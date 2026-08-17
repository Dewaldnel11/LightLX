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


_NARRATE = re.compile(
    r"\b(let me (read|check|look|open|inspect|see|add|implement|write|create|fix|update|modify|review|refactor|move|rename|delete|start|begin)|(now|next|then|first|finally),? (let me|i('ll| will)|i'm going to)|i('ll| will|'m going to|m going to) (read|check|look|open|add|implement|write|create|fix|update|modify|review|refactor|start|begin)|reading (the |more )?files|i('m| am) going to (read|check|look)|a subagent (is|was) (working|running)|(subagent|task) (is|was) (working|running|underway)|i (am|have) (awaiting|waiting for) (reports?|results?)|(the|this|my) (subagent|task) (is|has) (completed|finished|done|started)|parallel subagents? (are|were) (working|running)|(implementation|work|audit) (is|was) (currently )?(underway|in progress)|i (have|'ve) (applied|made|implemented) (edits?|changes?|fixes?))",
    re.I,
)
_NUM_PLAN = re.compile(
    r"^(\d+[\.\)]\s+.*\.(py|swift|js|ts|json|md|toml|cfg|txt)\s*){2,}$",
    re.MULTILINE | re.I,
)


def collapse_repeats(text: str) -> str:
    text = text or ""
    lines, out, seen = [], [], set()
    for ln in text.splitlines():
        key = " ".join(ln.split()).lower()
        if not key:
            if out and out[-1] != "":
                out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln.rstrip())
    return "\n".join(out).strip()


def is_repeating(text: str, min_hits=3) -> bool:
    text = text or ""
    if len(text) < 120:
        return False
    tail = text[-240:]
    if text[:-80].count(tail[:80]) >= min_hits:
        return True
    lines = [ln.strip().lower() for ln in text.splitlines() if len(ln.strip()) > 20]
    if not lines:
        return False
    last = lines[-1]
    return lines.count(last) >= min_hits + 1


def looks_like_tool_narration(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 1200:
        return False
    if _NARRATE.search(t):
        return True
    if _NUM_PLAN.search(t) and len(t) < 600:
        return True
    return False


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
