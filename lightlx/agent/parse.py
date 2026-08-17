import json
import re
import uuid

from .types import ToolCall

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:tool_call|json)\s*(\{.*?\})\s*```", re.DOTALL)
_QWEN = re.compile(
    r"<tool_call>\s*([A-Za-z0-9_\-\.]+)\s*(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_ARG = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)


def _args(raw):
    if isinstance(raw, dict):
        return raw
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"value": v}
    except json.JSONDecodeError:
        return {"_raw": s}


def _call(name, arguments, raw=""):
    return ToolCall(
        id="call_" + uuid.uuid4().hex[:12],
        name=(name or "").strip(),
        arguments=_args(arguments),
        raw_arguments=raw if isinstance(raw, str) else json.dumps(arguments or {}),
    )


def parse_text_tool_calls(text: str):
    text = text or ""
    calls = []
    used = []

    for m in _BLOCK.finditer(text):
        body = m.group(1).strip()
        if body.startswith("{") or body.startswith("["):
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict) and (data.get("name") or data.get("tool")):
                name = data.get("name") or data.get("tool")
                args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
                calls.append(_call(name, args, body))
                used.append(m.span())
                continue
        qm = _QWEN.match(m.group(0))
        if qm and not body.startswith("{"):
            name = qm.group(1)
            args = {k.strip(): v.strip() for k, v in _ARG.findall(qm.group(2))}
            if name:
                calls.append(_call(name, args, qm.group(2)))
                used.append(m.span())

    if not calls:
        for m in _FENCE.finditer(text):
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and (data.get("name") or data.get("tool")):
                name = data.get("name") or data.get("tool")
                args = data.get("arguments") or data.get("parameters") or {}
                calls.append(_call(name, args, m.group(1)))
                used.append(m.span())

    if not used:
        return text.strip(), calls
    cleaned = text
    for a, b in sorted(used, reverse=True):
        cleaned = cleaned[:a] + cleaned[b:]
    return cleaned.strip(), [c for c in calls if c.name]
