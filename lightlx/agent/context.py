import json
import re

from .providers import http_json, probe

DEFAULT_CONTEXT = 8192
COMPACT_RATIO = 0.72
KEEP_RECENT = 6

COMPACT_SYS = (
    "Summarize this conversation for a compact handoff. Preserve: the user's goals "
    "and constraints, files touched and what changed, decisions, errors, and what "
    "is left to do. Be dense. Do not continue the task — only summarize."
)


def estimate_tokens(messages) -> int:
    n = 0
    for m in messages or []:
        n += 8
        c = m.get("content")
        if isinstance(c, str):
            n += max(1, (len(c) + 3) // 4)
        elif c:
            n += max(1, (len(json.dumps(c, default=str)) + 3) // 4)
        if m.get("tool_calls"):
            n += max(1, (len(json.dumps(m["tool_calls"], default=str)) + 3) // 4)
    return n


def _intish(*vals):
    for v in vals:
        if v is None:
            continue
        try:
            n = int(v)
            if n > 0:
                return n
        except (TypeError, ValueError):
            continue
    return None


def _from_model_obj(m):
    if not isinstance(m, dict):
        return None
    return _intish(
        m.get("loaded_context_length"),
        m.get("max_context_length"),
        m.get("max_model_len"),
        m.get("context_length"),
        m.get("context_window"),
        m.get("max_context"),
        (m.get("meta") or {}).get("max_context_length") if isinstance(m.get("meta"), dict) else None,
    )


def detect_ollama_context(base, model):
    ps = (probe((base or "").rstrip("/") + "/api/ps") or {}).get("data") or {}
    for m in ps.get("models") or []:
        name = m.get("name") or m.get("model") or ""
        if name != model and not name.startswith(model) and model not in name:
            continue
        n = _intish(m.get("context_length"), m.get("context"), (m.get("details") or {}).get("context_length"))
        if n:
            return n
    try:
        data, _ = http_json("POST", (base or "").rstrip("/") + "/api/show", {"name": model}, timeout=8)
    except Exception:
        return DEFAULT_CONTEXT
    params = data.get("parameters") or ""
    for line in str(params).splitlines():
        if re.search(r"\bnum_ctx\b", line):
            n = _intish(line.split()[-1])
            if n:
                return n
    info = data.get("model_info") or {}
    if isinstance(info, dict):
        for k, v in info.items():
            if str(k).endswith("context_length") or str(k) in ("context_length", "max_position_embeddings"):
                n = _intish(v)
                if n:
                    return n
    return DEFAULT_CONTEXT


def detect_openai_context(base, model, api_key=None):
    from .providers import _headers
    for path in ("/api/v0/models", "/v1/models"):
        try:
            data, _ = http_json("GET", (base or "").rstrip("/") + path, headers=_headers(api_key), timeout=5)
        except Exception:
            data = (probe((base or "").rstrip("/") + path, api_key=api_key) or {}).get("data")
        if not data:
            continue
        rows = data.get("data") or data.get("models") or []
        hit = None
        for m in rows:
            mid = m.get("id") or m.get("name") or ""
            if mid == model or model in mid:
                n = _from_model_obj(m)
                if n:
                    return n
                hit = m
        if hit:
            n = _from_model_obj(hit)
            if n:
                return n
        if len(rows) == 1:
            n = _from_model_obj(rows[0])
            if n:
                return n
    return DEFAULT_CONTEXT


def detect_context(provider) -> int:
    known = getattr(provider, "context_length", None)
    if known:
        return int(known)
    kind = getattr(provider, "kind", "")
    base = getattr(provider, "base_url", "")
    model = getattr(provider, "model", "") or getattr(provider, "label", "")
    if kind == "mlx":
        n = int(getattr(provider, "ctx_limit", DEFAULT_CONTEXT) or DEFAULT_CONTEXT)
    elif kind == "ollama":
        n = detect_ollama_context(base, model)
    else:
        n = detect_openai_context(base, model, getattr(provider, "api_key", None))
    try:
        provider.context_length = n
    except Exception:
        pass
    return n


def room_for(context_length, max_tokens) -> int:
    ctx = max(int(context_length or DEFAULT_CONTEXT), 1024)
    reply = max(int(max_tokens or 0), 256)
    return max(ctx - reply - 256, 512)


def needs_compact(messages, context_length, max_tokens, ratio=COMPACT_RATIO) -> bool:
    return estimate_tokens(messages) > int(room_for(context_length, max_tokens) * ratio)


def _blob(messages) -> str:
    parts = []
    for m in messages:
        role = m.get("role") or "?"
        content = m.get("content") or ""
        if m.get("tool_calls") and not content:
            names = []
            for tc in m["tool_calls"]:
                names.append(getattr(tc, "name", None) or (tc.get("function") or {}).get("name") or "?")
            content = "tool_calls: " + ", ".join(names)
        if role == "tool":
            role = "tool:" + (m.get("name") or "")
        if len(content) > 4000:
            content = content[:4000] + "…"
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def compact_messages(provider, messages, *, keep=KEEP_RECENT, max_tokens=1024, on_event=None):
    if len(messages) <= keep + 1:
        return messages, False
    head = []
    rest = list(messages)
    if rest and rest[0].get("role") == "system":
        head = [rest.pop(0)]
    if len(rest) <= keep:
        return messages, False
    old, recent = rest[:-keep], rest[-keep:]
    if on_event:
        on_event("compact", detail=f"{estimate_tokens(old)} tok → summary")
    try:
        comp = provider.complete(
            [
                {"role": "system", "content": COMPACT_SYS},
                {"role": "user", "content": _blob(old)},
            ],
            tools=None,
            max_tokens=min(1500, max(400, max_tokens)),
            temperature=0.2,
        )
        summary = (comp.content or "").strip() or "(empty summary)"
    except Exception as e:
        summary = f"(compact failed: {e}; older turns dropped)"
    compacted = head + [
        {"role": "user", "content": "[compacted earlier conversation]\n" + summary},
        {"role": "assistant", "content": "Got it. I have the compacted context and will continue."},
    ] + recent
    return compacted, True


def maybe_compact(provider, messages, context_length, max_tokens, on_event=None, force=False):
    if not force and not needs_compact(messages, context_length, max_tokens):
        return messages, False
    return compact_messages(provider, messages, max_tokens=min(1200, max_tokens or 1024), on_event=on_event)


def handoff_note(old_label, new_label, context_length) -> dict:
    return {
        "role": "user",
        "content": (
            f"[handoff] Continuing this session on {new_label}"
            + (f" (was {old_label})" if old_label and old_label != new_label else "")
            + f". Context window is {context_length} tokens. "
            "Pick up from the compacted or recent conversation. Do not greet — continue the work."
        ),
    }
