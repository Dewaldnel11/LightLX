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
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(x) for x in obj]
    if hasattr(obj, "name") and hasattr(obj, "arguments"):
        return {
            "id": getattr(obj, "id", "") or "",
            "type": "function",
            "name": obj.name,
            "arguments": obj.arguments if isinstance(obj.arguments, dict) else {},
            "raw_arguments": getattr(obj, "raw_arguments", "") or "",
        }
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
    key = project_key(record)
    if key:
        for rec in list_sessions(200, one_per_project=False):
            other = rec.get("id")
            if other and other != sid and project_key(rec) == key:
                delete_session(other)
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


def project_key(rec) -> str:
    ws = (rec or {}).get("workspace") or ""
    if ws:
        try:
            return os.path.abspath(os.path.expanduser(str(ws))).lower()
        except Exception:
            return str(ws).strip().lower()
    return str((rec or {}).get("id") or "")


def project_name(rec) -> str:
    ws = (rec or {}).get("workspace") or ""
    name = os.path.basename(str(ws).rstrip("/")) if ws else ""
    return name or (rec or {}).get("title") or (rec or {}).get("id") or "project"


def list_sessions(limit=20, one_per_project=False):
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
    if one_per_project:
        seen, uniq = set(), []
        for d in rows:
            k = project_key(d)
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(d)
        rows = uniq
    return rows[:limit]


def delete_session(sid: str) -> bool:
    p = path_for(sid)
    try:
        os.remove(p)
        return True
    except Exception:
        return False


def hydrate_messages(messages):
    from .types import ToolCall
    from .providers import parse_args
    out = []
    for raw in messages or []:
        m = dict(raw)
        tcs = m.get("tool_calls")
        if tcs:
            calls = []
            for tc in tcs:
                if isinstance(tc, ToolCall):
                    calls.append(tc)
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
                raw_a = fn.get("raw_arguments") if isinstance(fn, dict) else tc.get("raw_arguments")
                if raw_a is None:
                    raw_a = args if isinstance(args, str) else ""
                name = (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name") or ""
                calls.append(ToolCall(
                    id=tc.get("id") or "",
                    name=name,
                    arguments=parse_args(args),
                    raw_arguments=raw_a if isinstance(raw_a, str) else "",
                ))
            m["tool_calls"] = calls
        out.append(m)
    return out


def title_from(history) -> str:
    for m in history or []:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().splitlines()[0]
            if t.startswith("["):
                continue
            return (t[:72] + "…") if len(t) > 72 else t
    return "untitled"


def _call_name(tc):
    if hasattr(tc, "name"):
        return tc.name or ""
    if isinstance(tc, dict):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        return (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name") or ""
    return ""


def _call_id(tc):
    if hasattr(tc, "id"):
        return tc.id or ""
    if isinstance(tc, dict):
        return tc.get("id") or ""
    return ""


def history_for_save(messages):
    msgs = list(messages or [])
    out = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
            names = {_call_name(tc) for tc in calls}
            keep = "task" in names
            results, j = [], i + 1
            while j < len(msgs) and msgs[j].get("role") == "tool":
                results.append(msgs[j])
                j += 1
            if keep:
                out.append(m)
                for r in results:
                    rr = dict(r)
                    is_task = rr.get("name") == "task" or rr.get("tool_call_id") in {
                        _call_id(tc) for tc in calls if _call_name(tc) == "task"
                    }
                    content = str(rr.get("content") or "")
                    if is_task:
                        if len(content) > 4000:
                            rr["content"] = content[:4000] + "\n… truncated"
                    elif len(content) > 120:
                        rr["content"] = "(elided)"
                    out.append(rr)
            else:
                stripped = dict(m)
                stripped.pop("tool_calls", None)
                out.append(stripped)
            i = j
            continue
        if role in ("user", "assistant", "system"):
            out.append(m)
        i += 1
    return out


def id_for_workspace(workspace, current=None):
    if current:
        return current
    key = project_key({"workspace": workspace})
    if not key:
        return new_id()
    for rec in list_sessions(200, one_per_project=False):
        if project_key(rec) == key:
            return rec.get("id") or new_id()
    return new_id()


def record_from(sess, source=None):
    hist = history_for_save(
        [m for m in (sess.history or []) if m.get("role") in ("user", "assistant", "system", "tool")]
    )
    src = source or getattr(sess, "source", None) or {}
    pending = getattr(sess, "pending", None)
    ws = str(sess.ws.root)
    return {
        "id": id_for_workspace(ws, getattr(sess, "session_id", None)),
        "title": title_from(hist) or title_from(pending),
        "source": src,
        "workspace": ws,
        "history": hist,
        "pending": pending or None,
        "in_progress": bool(pending),
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


def idle_session_paths(idle_seconds, now=None, sess_dir=None):
    now = now if now is not None else time.time()
    folder = sess_dir or SESS_DIR
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        p = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(p) >= int(idle_seconds or 900):
                out.append(p)
        except Exception:
            continue
    return out
