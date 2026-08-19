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
    args = _args(arguments)
    raw_str = raw if isinstance(raw, str) else ""
    if raw_str:
        try:
            json.loads(raw_str)
        except Exception:
            raw_str = json.dumps(args)
    else:
        raw_str = json.dumps(args)
    return ToolCall(
        id="call_" + uuid.uuid4().hex[:12],
        name=name,
        arguments=args,
        raw_arguments=raw_str,
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


# Recovery for weak local models that emit a tool call as bare text like
#   write_file path=/a/b.yml content=<multiline body>
# often wrapped in a ```bash fence and trailed by stray </parameter></function>
# </tool_call> tags. None of the structured parsers above catch this, so the
# file silently never gets written.
_KV_TOOL_NAMES = (
    "read_file", "write_file", "edit_file", "bash", "list_dir", "glob", "grep",
    "fetch_url", "web_search", "docs", "memory", "brain_search", "brain_write",
    "task", "skill",
)
_KV_NAME_ALT = "|".join(_KV_TOOL_NAMES)
_KV_START = re.compile(rf"(?im)^[ \t]*({_KV_NAME_ALT})[ \t]+(?=[a-z_]+[ \t]*=)")
_KV_STRAY_TAG = re.compile(
    r"</?(parameter|function|tool_call|arg_value|arg_key|invoke)>", re.I
)
_KV_ARG_KEY = re.compile(
    r"(?:^|[\s,])(path|file_path|file|content|command|cmd|old_string|new_string|"
    r"old_str|new_str|pattern|query|url|offset|limit|glob|description|"
    r"subagent_type|prompt|workdir|body|text|code|script)[ \t]*=",
    re.I,
)
_KV_SINK = {
    "content", "command", "cmd", "old_string", "new_string", "old_str",
    "new_str", "body", "text", "code", "script", "prompt",
}


def _clean_kv_value(v, sink):
    v = v.strip("\n")
    if not sink:
        return v.strip().rstrip(",").strip("\"'")
    v = _KV_STRAY_TAG.sub("", v).strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        v = v[1:-1]
    return v


def _parse_kv_args(blob):
    keys = list(_KV_ARG_KEY.finditer(blob))
    if not keys:
        return {}
    args = {}
    for i, km in enumerate(keys):
        key = km.group(1).lower()
        vstart = km.end()
        if key in _KV_SINK:
            args[key] = _clean_kv_value(blob[vstart:], sink=True)
            break
        vend = keys[i + 1].start() if i + 1 < len(keys) else len(blob)
        args[key] = _clean_kv_value(blob[vstart:vend], sink=False)
    return args


# Live-stream gating: a tool call narrated as text (```bash\nwrite_file path=…)
# should never be dumped to the terminal. The gate flushes genuine prose but,
# the moment it recognizes a tool-call opener, stops emitting and signals the UI
# to show a spinner instead. The real call is summarized later on execution.
_STREAM_OPENER = re.compile(
    rf"(?im)^[ \t]*(?:`{{3}}(?:bash|sh|shell|zsh|console|terminal)[ \t]*\n"
    rf"|`{{3}}[\w+-]*[ \t]*\n)?[ \t]*"
    rf"(?:<tool_call>|<function|<arg_key>|(?:{_KV_NAME_ALT})[ \t]+[a-z_]+[ \t]*=)"
)
_FENCE_OPEN = re.compile(r"^[ \t]*`{3}[\w+-]*[ \t]*$")
_FENCE_CLOSE = re.compile(r"^[ \t]*`{3}[ \t]*$")


class StreamGate:
    """Buffers streamed text, emitting prose but hiding tool-call narration."""

    def __init__(self, emit, on_suppress=None):
        self._emit = emit
        self._on_suppress = on_suppress
        self.suppressed = False
        self._buf = ""
        self._fence = None  # held fence lines awaiting a verdict, or None
        self._fence_shell = False

    def _suppress(self):
        if not self.suppressed:
            self.suppressed = True
            self._fence = None
            self._fence_shell = False
            self._buf = ""
            if self._on_suppress:
                self._on_suppress()

    def feed(self, piece):
        if self.suppressed or not piece:
            return
        self._buf += piece
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line = self._buf[: nl + 1]
            self._buf = self._buf[nl + 1:]
            self._line(line)
            if self.suppressed:
                return
        probe = ("".join(self._fence) if self._fence else "") + self._buf
        if probe and _STREAM_OPENER.search(probe):
            self._suppress()

    def _line(self, line):
        stripped = line.rstrip("\n")
        if self._fence is not None:
            if self._fence_shell:
                if _FENCE_CLOSE.match(stripped):
                    self._suppress()
                return
            if _STREAM_OPENER.search("".join(self._fence) + line):
                self._suppress()
                return
            if _FENCE_CLOSE.match(stripped) or stripped.strip():
                self._emit("".join(self._fence) + line)
                self._fence = None
                return
            self._fence.append(line)
            return
        if _STREAM_OPENER.search(line):
            self._suppress()
            return
        if _FENCE_OPEN.match(stripped):
            self._fence = [line]
            self._fence_shell = bool(_SHELL_FENCE_LANG.match(stripped))
            return
        self._emit(line)

    def close(self):
        if self.suppressed:
            return
        if self._fence_shell:
            self._suppress()
            return
        if self._fence is not None:
            self._emit("".join(self._fence))
            self._fence = None
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


_SHELL_FENCE = re.compile(
    r"```(?:bash|sh|shell|zsh|console|terminal)\s*\n(.*?)```",
    re.DOTALL | re.I,
)
_SHELL_FENCE_LANG = re.compile(
    r"^[ \t]*```(?:bash|sh|shell|zsh|console|terminal)[ \t]*$",
    re.I,
)


def fenced_shell_calls(text):
    # Models often dump runnable shell as ```bash\nfind …\n``` instead of calling bash.
    calls = []
    for m in _SHELL_FENCE.finditer(text or ""):
        body = (m.group(1) or "").strip()
        if not body or _KV_START.search(body):
            continue
        calls.append(_call("bash", {"command": body}, body))
    return calls


def kv_tool_calls(text):
    # Drop fence lines and stray closing tags but keep body indentation intact.
    lines = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("```"):
            continue
        if _KV_STRAY_TAG.fullmatch(s):
            continue
        lines.append(ln)
    blob = "\n".join(lines)
    starts = list(_KV_START.finditer(blob))
    calls = []
    for i, sm in enumerate(starts):
        name = sm.group(1)
        seg_start = sm.end()
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(blob)
        args = _parse_kv_args(blob[seg_start:seg_end])
        if args:
            calls.append(_call(name, args, blob[seg_start:seg_end]))
    return calls


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
    if _SHELL_FENCE.search(t):
        return True
    if _NARRATE.search(t):
        return True
    if _NUM_PLAN.search(t) and len(t) < 600:
        return True
    return False


_PROMISE = re.compile(
    r"\b((now|next|then|first|finally),?\s+)?(i('ll| will)|let me|i am going to|i'm going to)\s+"
    r"(writ\w*|add\w*|implement\w*|creat\w*|edit\w*|updat\w*|fix\w*|mak\w*|dump\w*|"
    r"sav\w*|check\w*|search\w*|look\w*|start\w*|continu\w*|finish\w*)",
    re.I,
)
_DONE_CLAIM = re.compile(
    r"\b(done\.|that's (it|all)|task (is )?(complete|done)|implement(ed|ation)|finished the|"
    r"all (set|good|done)|changes (are|have been) (in|applied))\b",
    re.I,
)
# Stems (implement\w* etc.) so "implementing", "writing", "fixes" all match —
# a bare \bimplement\b misses "implementing", which silently disabled the loop.
_MUTATION = re.compile(
    r"\b(implement\w*|fix(e[sd]|ing)?|add(s|ing|ed)?|writ(e|es|ing)|edit\w*|updat\w*|"
    r"refactor\w*|remov\w*|delet\w*|creat\w*|patch\w*|integrat\w*|"
    r"make (the )?(change|edit|fix|patch)\w*|do the work|build (out|the))\b",
    re.I,
)
_QUESTION_LEAD = re.compile(r"^(how|what|why|explain|status|when|where|who)\b", re.I)
_FENCE_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_TOOL_LANG = {"tool_call", "tool_code"}
_QUOTED_PATH = re.compile(
    r"[`'\"]((?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,8})[`'\"]"
)
_BARE_PATH = re.compile(
    r"(?<![\w./])((?:[\w.-]+/)+[\w.-]+\.(?:yml|yaml|py|swift|js|ts|tsx|json|md|toml|sh|txt|cfg))"
)


def _norm_path(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lower()


def is_implementation_request(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if "kickoff protocol" in low or low.startswith("follow the kickoff"):
        return False
    if re.search(
        r"\b(resume|continue|finish)\w*\b.*\b(work|kickoff|plan|implement\w*|files?)\b",
        low,
    ) or re.search(
        r"\b(kickoff|previous)\b.*\b(work|plan|implement\w*)\b",
        low,
    ):
        return True
    if _QUESTION_LEAD.match(t) and not _MUTATION.search(t):
        return False
    if t.rstrip().endswith("?") and not _MUTATION.search(t):
        return False
    return bool(_MUTATION.search(t))


def completion_text_signals(text: str) -> dict:
    t = text or ""
    announced = set()
    for m in _QUOTED_PATH.finditer(t):
        announced.add(_norm_path(m.group(1)))
    for m in _BARE_PATH.finditer(t):
        announced.add(_norm_path(m.group(1)))
    unapplied = False
    for lang, body in _FENCE_BLOCK.findall(t):
        kind = (lang or "").strip().split()[0].lower() if (lang or "").strip() else ""
        if kind in _TOOL_LANG:
            continue
        blob = (body or "").strip()
        if "<tool_call>" in blob or "```tool_code" in blob:
            continue
        if len(blob) >= 40:
            unapplied = True
            break
    return {
        "action_promise": bool(_PROMISE.search(t) or _NARRATE.search(t)),
        "unapplied_code": unapplied,
        "done_claim": bool(_DONE_CLAIM.search(t)),
        "announced_paths": announced,
    }


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

    if not calls:
        kv = kv_tool_calls(text)
        if kv:
            calls.extend(kv)
            first = _KV_START.search(text)
            fence = None
            if first:
                head = text[: first.start()]
                fm = re.search(r"```[^\n`]*\s*$", head)
                fence = fm.start() if fm else first.start()
            used.append((fence if fence is not None else 0, len(text)))

    if not calls:
        shell = fenced_shell_calls(text)
        if shell:
            calls.extend(shell)
            for m in _SHELL_FENCE.finditer(text):
                used.append(m.span())
                break

    if not used:
        return text.strip(), calls
    cleaned = text
    for a, b in sorted(used, reverse=True):
        cleaned = cleaned[:a] + cleaned[b:]
    return cleaned.strip(), [c for c in calls if c.name]
