import ast
import json
import re
import uuid

from .types import ToolCall

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:tool_call|tool_code|json)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_QWEN = re.compile(
    r"<tool_call>\s*([A-Za-z0-9_\-\.]+)\s*(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_ARG = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)
_FUNC_XML = re.compile(
    r"<function\s*[=:]\s*([A-Za-z0-9_\-\.]+)\s*>(.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_INVOKE = re.compile(
    r"invoke\s+(?:tool\s+)?([A-Za-z0-9_\-\.]+)\s+with\s+(.+)",
    re.IGNORECASE,
)
_KV = re.compile(r"([A-Za-z_][\w]*)\s*(?:=|:|is)\s*(\"[^\"]*\"|'[^']*'|[^\s,]+)", re.I)
_PY_SKIP = {
    "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple",
    "open", "format", "type", "min", "max", "sum", "sorted", "enumerate", "zip",
    "map", "filter", "bool", "repr", "abs", "round", "any", "all", "super",
    "isinstance", "hasattr", "getattr", "setattr", "callable", "iter", "next",
}


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
    name = (name or "").strip()
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return ToolCall(
        id="call_" + uuid.uuid4().hex[:12],
        name=name,
        arguments=_args(arguments),
        raw_arguments=raw if isinstance(raw, str) else json.dumps(arguments or {}),
    )


def _from_mapping(data, raw=""):
    if isinstance(data, list):
        out = []
        for item in data:
            out.extend(_from_mapping(item, raw))
        return out
    if not isinstance(data, dict):
        return []
    name = data.get("name") or data.get("tool") or data.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    if not name:
        return []
    args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
    return [_call(name, args, raw or json.dumps(data))]


def _py_kwargs(raw):
    raw = (raw or "").strip().rstrip(",")
    if not raw:
        return {}
    try:
        tree = ast.parse(f"f({raw})", mode="eval")
    except SyntaxError:
        return {}
    if not isinstance(tree.body, ast.Call):
        return {}
    out = {}
    for kw in tree.body.keywords:
        if not kw.arg:
            continue
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            out[kw.arg] = ast.unparse(kw.value) if hasattr(ast, "unparse") else ""
    return out


def _split_call_args(text, open_idx):
    depth = 0
    quote = None
    escape = False
    for i in range(open_idx, len(text)):
        c = text[i]
        if quote:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                quote = None
            continue
        if c in ('"', "'"):
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i + 1
    return None, None


def extract_python_tool_calls(text):
    calls, used = [], []
    i = 0
    ident = re.compile(r"[A-Za-z_][\w\.]*")
    while i < len(text):
        m = ident.search(text, i)
        if not m:
            break
        name = m.group(0)
        j = m.end()
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "(":
            i = m.end()
            continue
        args_s, end = _split_call_args(text, j)
        if args_s is None:
            break
        base = name.rsplit(".", 1)[-1]
        if base.lower() in _PY_SKIP:
            i = j + 1
            continue
        kwargs = _py_kwargs(args_s)
        if kwargs or args_s.strip() == "":
            calls.append(_call(base, kwargs, args_s))
            used.append((m.start(), end))
        i = end
    return calls, used


def _invoke_args(blob):
    out = {}
    for k, v in _KV.findall(blob or ""):
        out[k] = v.strip().strip("\"'")
    return out


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
            found = _from_mapping(data, body) if data is not None else []
            if found:
                calls.extend(found)
                used.append(m.span())
                continue
        qm = _QWEN.match(m.group(0))
        if qm and not body.startswith("{"):
            name = qm.group(1)
            args = {k.strip(): v.strip() for k, v in _ARG.findall(qm.group(2))}
            if name:
                calls.append(_call(name, args, qm.group(2)))
                used.append(m.span())

    for m in _FUNC_XML.finditer(text):
        raw = m.group(2).strip()
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = _invoke_args(raw)
        if isinstance(data, dict):
            calls.append(_call(m.group(1), data, raw))
            used.append(m.span())

    if not calls:
        for m in _FENCE.finditer(text):
            body = (m.group(1) or "").strip()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
            found = _from_mapping(data, body) if data is not None else []
            if found:
                calls.extend(found)
                used.append(m.span())
                continue
            py, _ = extract_python_tool_calls(body)
            if py:
                calls.extend(py)
                used.append(m.span())

    if not calls:
        for m in _INVOKE.finditer(text):
            calls.append(_call(m.group(1), _invoke_args(m.group(2)), m.group(2)))
            used.append(m.span())

    if not used:
        return text.strip(), calls
    cleaned = text
    for a, b in sorted(used, reverse=True):
        cleaned = cleaned[:a] + cleaned[b:]
    return cleaned.strip(), [c for c in calls if c.name]
