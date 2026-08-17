import json
import os
import time
import uuid
from datetime import datetime, timezone

from ..state import STATE_DIR

SESS_DIR = os.path.join(STATE_DIR, "sessions")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(obj):
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items() if k != "tool_calls" or isinstance(v, (list, dict, str, int, float, bool, type(None)))}
    if isinstance(obj, list):
        out = []
        for x in obj:
            if hasattr(x, "name") and hasattr(x, "arguments"):
                out.append({"name": x.name, "arguments": x.arguments, "id": getattr(x, "id", "")})
            else:
                out.append(_safe(x))
        return out
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def new_id():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def path_for(sid):
    return os.path.join(SESS_DIR, f"{sid}.json")


def save_session(record: dict) -> str:
    sid = record.get("id") or new_id()
    record["id"] = sid
    record["updated"] = _now()
    record.setdefault("created", record["updated"])
    os.makedirs(SESS_DIR, exist_ok=True)
    with open(path_for(sid), "w") as f:
        json.dump(_safe(record), f, indent=2)
    return sid


def load_session(sid: str) -> dict | None:
    if not sid:
        return None
    p = path_for(sid)
    if not os.path.isfile(p) and not sid.endswith(".json"):
        alt = os.path.join(SESS_DIR, sid)
        p = alt if os.path.isfile(alt) else p
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def list_sessions(limit=20):
    if not os.path.isdir(SESS_DIR):
        return []
    rows = []
    for name in os.listdir(SESS_DIR):
        if not name.endswith(".json"):
            continue
        p = os.path.join(SESS_DIR, name)
        try:
            with open(p) as f:
                d = json.load(f)
            rows.append(d)
        except Exception:
            continue
    rows.sort(key=lambda d: d.get("updated") or d.get("created") or "", reverse=True)
    return rows[:limit]


def delete_session(sid: str) -> bool:
    p = path_for(sid)
    try:
        os.remove(p)
        return True
    except Exception:
        return False


def title_from(history) -> str:
    for m in history or []:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().splitlines()[0]
            if t.startswith("["):
                continue
            return (t[:72] + "…") if len(t) > 72 else t
    return "untitled"


def record_from(sess, source=None):
    hist = [m for m in (sess.history or []) if m.get("role") in ("user", "assistant", "system")]
    src = source or getattr(sess, "source", None) or {}
    return {
        "id": getattr(sess, "session_id", None) or new_id(),
        "title": title_from(hist),
        "source": src,
        "workspace": str(sess.ws.root),
        "history": hist,
        "provider": getattr(sess.provider, "label", ""),
        "kind": getattr(sess.provider, "kind", ""),
        "context_length": getattr(sess.provider, "context_length", None),
        "max_tokens": sess.max_tokens,
    }


def age(updated):
    if not updated:
        return ""
    try:
        ts = datetime.strptime(updated.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        sec = max(0, int((datetime.now(timezone.utc) - ts).total_seconds()))
    except Exception:
        return updated
    if sec < 60:
        return "just now"
    if sec < 3600:
        return f"{sec // 60}m ago"
    if sec < 86400:
        return f"{sec // 3600}h ago"
    return f"{sec // 86400}d ago"
