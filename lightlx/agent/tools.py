import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from .prompts import DOC_ALIASES
from .types import ToolSpec

MAX_READ = 200_000
MAX_LINES = 2000
MAX_TOOL_OUT = 80_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mlx", ".cache"}


class Workspace:
    def __init__(self, root):
        self.root = Path(root or os.getcwd()).expanduser().resolve()

    def resolve(self, path):
        p = Path(path or "").expanduser()
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    def rel(self, path):
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self.root))
        except Exception:
            return str(p)


def _clip(text, limit=MAX_TOOL_OUT):
    text = text if isinstance(text, str) else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… truncated ({len(text)} chars total)"


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except Exception:
        return False


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTML()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def http_get(url, timeout=30, max_bytes=1_500_000):
    req = Request(url, headers={"User-Agent": "lightlx/0.2.1", "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read(max_bytes)
        ctype = r.headers.get("Content-Type", "")
        final = r.geturl()
    return data, ctype, final


class BuiltinTools:
    def __init__(self, workspace: Workspace, launch_task=None, skills=None, memory=None):
        self.ws = workspace
        self.launch_task = launch_task
        self.skills = skills or {}
        self.memory = memory

    def specs(self, include_task=True, readonly=False):
        out = [
            self._spec("read_file", "Read a file. Optional offset/limit are 1-indexed line numbers.", {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            }, self.read_file),
            self._spec("list_dir", "List a directory (names, types, sizes).", {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory, default workspace root"},
                    "depth": {"type": "integer", "description": "1 = this folder only"},
                },
            }, self.list_dir),
            self._spec("glob", "Find files by glob pattern relative to workspace.", {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "e.g. **/*.py"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            }, self.glob),
            self._spec("grep", "Search file contents. Uses ripgrep when installed.", {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "File filter, e.g. *.py"},
                    "max_hits": {"type": "integer"},
                },
                "required": ["pattern"],
            }, self.grep),
            self._spec("fetch_url", "Fetch a URL and return text (HTML stripped).", {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["url"],
            }, self.fetch_url),
            self._spec("web_search", "Search the public web (DuckDuckGo). Use before fetch_url. Several searches in one turn run in parallel.", {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            }, self.web_search),
            self._spec("skill", "Load a Claude/Codex/LightLX skill. Skills marked context:fork run as a subagent. On LM Studio, do not fork several skills in one turn.", {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "string", "description": "Optional args passed to the skill"},
                    "fork": {"type": "boolean", "description": "Run as a subagent even if the skill is not marked fork"},
                },
                "required": ["name"],
            }, self.skill),
            self._spec("memory", "Read or write auto-memory (lasting notes across sessions). action=read|write|list. Keep MEMORY.md a short index.", {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "list"]},
                    "path": {"type": "string", "description": "MEMORY.md or a topic file like debugging.md"},
                    "content": {"type": "string", "description": "Full file contents when action=write"},
                },
                "required": ["action"],
            }, self.memory_tool),
            self._spec("brain_search", "Search the cross-project brain and all project memory files.", {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string", "description": "preference, correction, gotcha, workflow, project"},
                    "scope": {"type": "string"},
                },
                "required": ["query"],
            }, self.brain_search_tool),
            self._spec("brain_write", "Write a typed cross-project brain record. Web facts need url. Do not store secrets or unverified claims.", {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["preference", "correction", "gotcha", "workflow", "project"]},
                    "text": {"type": "string"},
                    "scope": {"type": "string"},
                    "source": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "url": {"type": "string", "description": "Primary source URL for web-grounded facts"},
                },
                "required": ["kind", "text"],
            }, self.brain_write_tool),
            self._spec("docs", "Read docs/source from Claude Code, Codex, Ollama, MCP, or any GitHub repo.", {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Alias (claude-code, codex, ollama, lmstudio, mcp, lightlx) or owner/repo",
                    },
                    "path": {"type": "string", "description": "File or directory in the repo. Default README.md"},
                    "ref": {"type": "string", "description": "Branch or tag. Tries main, then master."},
                    "list": {"type": "boolean", "description": "List a directory instead of reading a file"},
                },
                "required": ["source"],
            }, self.docs),
        ]
        if not readonly:
            out.extend([
                self._spec("write_file", "Create or overwrite a file. Creates parent folders.", {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                }, self.write_file),
                self._spec("edit_file", "Replace exact text in a file. old_string must be unique unless replace_all.", {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["path", "old_string", "new_string"],
                }, self.edit_file),
                self._spec("bash", "Run a shell command in the workspace. Returns stdout, stderr, exit code.", {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "description": "Seconds, default 60"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["command"],
                }, self.bash),
            ])
        if include_task and self.launch_task:
            out.append(self._spec(
                "task",
                "Launch a subagent for a big or long-horizon job. On LM Studio / a single local "
                "GPU, issue ONE task at a time — parallel task calls drop the parent stream. "
                "On backends that allow it, several task calls in the SAME turn run in parallel. "
                "explore = read-only research; implement = can edit files; general = full tools. "
                "Subagents cannot spawn further subagents. Tell them exactly what to return.",
                {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Short 3-8 word label"},
                        "prompt": {"type": "string", "description": "Full instructions and what to report back"},
                        "subagent_type": {
                            "type": "string",
                            "enum": ["explore", "implement", "general"],
                            "description": "explore (read-only), implement (edit+bash), general (default)",
                        },
                    },
                    "required": ["description", "prompt"],
                },
                self.task,
            ))
        return out

    def _spec(self, name, desc, params, handler):
        return ToolSpec(name, desc, params, handler)

    def read_file(self, path, offset=None, limit=None):
        p = self.ws.resolve(path)
        if not p.exists():
            return f"error: not found: {p}"
        if p.is_dir():
            return f"error: {p} is a directory — use list_dir"
        if _is_binary(p):
            return f"error: binary file ({p.stat().st_size} bytes)"
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        start = max(int(offset or 1), 1)
        n = int(limit or MAX_LINES)
        chunk = lines[start - 1:start - 1 + n]
        numbered = [f"{i + start}| {ln}" for i, ln in enumerate(chunk)]
        extra = ""
        if start - 1 + n < len(lines):
            extra = f"\n… {len(lines) - (start - 1 + n)} more lines (offset={start + n})"
        return _clip(f"{self.ws.rel(p)}  {len(lines)} lines\n" + "\n".join(numbered) + extra)

    def write_file(self, path, content):
        p = self.ws.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else str(content))
        return f"wrote {self.ws.rel(p)} ({len(content)} chars)"

    def edit_file(self, path, old_string, new_string, replace_all=False):
        p = self.ws.resolve(path)
        if not p.is_file():
            return f"error: not a file: {p}"
        text = p.read_text(errors="replace")
        count = text.count(old_string)
        if count == 0:
            return "error: old_string not found"
        if count > 1 and not replace_all:
            return f"error: old_string found {count} times — pass replace_all=true or more context"
        p.write_text(text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1))
        return f"edited {self.ws.rel(p)} ({count if replace_all else 1} replacement)"

    def list_dir(self, path=None, depth=1):
        p = self.ws.resolve(path or ".")
        if not p.exists():
            return f"error: not found: {p}"
        if p.is_file():
            return f"file {self.ws.rel(p)} {p.stat().st_size} bytes"
        depth = max(int(depth or 1), 1)
        lines = []

        def walk(cur, d, prefix=""):
            try:
                kids = sorted(cur.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                lines.append(prefix + "permission denied")
                return
            for k in kids:
                if k.name in SKIP_DIRS:
                    continue
                if k.is_dir():
                    lines.append(f"{prefix}{k.name}/")
                    if d > 1:
                        walk(k, d - 1, prefix + "  ")
                else:
                    try:
                        sz = k.stat().st_size
                    except OSError:
                        sz = 0
                    lines.append(f"{prefix}{k.name}  {sz}")

        walk(p, depth)
        header = str(self.ws.rel(p)) + "/\n"
        return _clip(header + "\n".join(lines) if lines else header + "(empty)")

    def glob(self, pattern, path=None):
        root = self.ws.resolve(path or ".")
        if not root.exists():
            return f"error: not found: {root}"
        hits = []
        for match in sorted(root.glob(pattern)):
            if any(part in SKIP_DIRS for part in match.parts):
                continue
            if match.is_file():
                hits.append(self.ws.rel(match))
            if len(hits) >= 400:
                break
        return _clip("\n".join(hits) if hits else "(no matches)")

    def grep(self, pattern, path=None, glob=None, max_hits=80):
        root = self.ws.resolve(path or ".")
        max_hits = int(max_hits or 80)
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "--line-number", "--no-heading", "--color", "never", "-m", str(max_hits), pattern]
            if glob:
                cmd.extend(["--glob", glob])
            cmd.append(str(root))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                out = r.stdout.strip() or r.stderr.strip() or "(no matches)"
                return _clip(out)
            except Exception:
                pass
        rx = re.compile(pattern)
        hits = []
        files = [root] if root.is_file() else root.rglob(glob or "*")
        for f in files:
            if not getattr(f, "is_file", lambda: False)():
                continue
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if _is_binary(f):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self.ws.rel(f)}:{i}:{line}")
                        if len(hits) >= max_hits:
                            return _clip("\n".join(hits))
            except Exception:
                continue
        return _clip("\n".join(hits) if hits else "(no matches)")

    def bash(self, command, timeout=60, workdir=None):
        cwd = self.ws.resolve(workdir) if workdir else self.ws.root
        if not cwd.is_dir():
            return f"error: workdir is not a directory: {cwd}"
        try:
            r = subprocess.run(
                command, shell=True, cwd=str(cwd), capture_output=True, text=True,
                timeout=int(timeout or 60), env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return f"error: timed out after {timeout}s"
        out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
        tail = f"\nexit {r.returncode}" if r.returncode else ""
        return _clip((out.strip() + tail) or f"(no output) exit {r.returncode}")

    def fetch_url(self, url, max_chars=20000):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "error: only http/https"
        try:
            data, ctype, final = http_get(url)
        except Exception as e:
            return f"error: {e}"
        text = data.decode("utf-8", "replace")
        if "html" in (ctype or "") or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            text = html_to_text(text)
        return _clip(f"url: {final}\n\n{text}", int(max_chars or 20000))

    def web_search(self, query, max_results=8):
        from .brain import parse_ddg_html
        q = (query or "").strip()
        if not q:
            return "error: query required"
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(q)
        try:
            data, ctype, final = http_get(url, timeout=20)
        except Exception as e:
            return f"error: {e}"
        html = data.decode("utf-8", "replace")
        rows = parse_ddg_html(html)[: int(max_results or 8)]
        if not rows:
            text = html_to_text(html)[:4000]
            return _clip(f"url: {final}\n(no structured results)\n{text}")
        lines = [f"url: {final}", f"query: {q}", ""]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}")
        return "\n".join(lines)

    def brain_search_tool(self, query, kind="", scope=""):
        from .brain import brain_search
        return brain_search(query, kind=kind or "", scope=scope or "")

    def brain_write_tool(self, kind, text, scope="global", source="user", confidence="high", url=""):
        from .brain import write_record
        if url and not str(url).startswith("http"):
            return "error: url must be http(s) — unverified claims are not stored"
        conf = confidence or "high"
        if not url and conf != "high":
            return "error: web/idle facts need a source url or confidence=high from the user"
        return write_record(
            kind, text, scope=scope or "global", source=source or "user",
            confidence=conf, url=url or "",
        )

    def docs(self, source, path=None, ref=None, list=False):
        repo = DOC_ALIASES.get((source or "").strip().lower(), source.strip())
        if "/" not in repo:
            known = ", ".join(sorted(set(DOC_ALIASES)))
            return f"error: unknown source {source!r}. Use owner/repo or one of: {known}"
        path = (path or "README.md").lstrip("/")
        refs = [ref] if ref else ["main", "master"]
        last_err = None
        for branch in refs:
            if list:
                url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
                try:
                    data, _, _ = http_get(url)
                    items = json.loads(data.decode())
                except Exception as e:
                    last_err = e
                    continue
                if isinstance(items, dict) and items.get("message"):
                    last_err = items.get("message")
                    continue
                if isinstance(items, dict):
                    return f"{repo}@{branch}:{path}  (file — set list=false to read)"
                lines = []
                for it in items:
                    mark = "/" if it.get("type") == "dir" else ""
                    lines.append(f"{it.get('name')}{mark}")
                return f"{repo}@{branch}:{path}\n" + "\n".join(lines)
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            try:
                data, ctype, final = http_get(url)
            except Exception as e:
                last_err = e
                continue
            if data.startswith(b"404") or b"Not Found" in data[:40]:
                last_err = "not found"
                continue
            text = data.decode("utf-8", "replace")
            return _clip(f"{repo}@{branch}:{path}\n{final}\n\n{text}")
        return f"error: could not read {repo}:{path} ({last_err})"

    def task(self, description, prompt, subagent_type="general"):
        if not self.launch_task:
            return "error: subagents are not available"
        return self.launch_task(description or "task", prompt or "", subagent_type or "general")

    def skill(self, name, arguments="", fork=None):
        from .skills import expand_skill
        key = (name or "").strip().lstrip("/")
        sk = self.skills.get(key) or self.skills.get(key.lower())
        if sk is None:
            known = ", ".join(sorted(self.skills)) or "(none)"
            return f"error: unknown skill {name!r}. available: {known}"
        body = expand_skill(sk, arguments or "", workspace=str(self.ws.root))
        want_fork = fork if fork is not None else sk.fork
        if isinstance(want_fork, str):
            want_fork = want_fork.lower() in ("true", "1", "yes", "on")
        if want_fork and self.launch_task:
            return self.launch_task(sk.name, body, sk.agent_kind)
        if want_fork:
            return "(subagents not available — run this skill inline)\n\n" + body
        return body

    def memory_tool(self, action, path="MEMORY.md", content=""):
        if not self.memory:
            return "error: memory is not available"
        act = (action or "read").lower()
        if act == "list":
            files = self.memory.list_files()
            return "\n".join(files) if files else "(empty — write MEMORY.md to start)"
        if act == "read":
            return self.memory.read(path or "MEMORY.md")
        if act == "write":
            if not content:
                return "error: content required for write"
            return self.memory.write(path or "MEMORY.md", content)
        return "error: action must be read, write, or list"


def _short_label(text, n=40):
    s = str(text or "").strip()
    if "  " in s:
        s = s.split("  ", 1)[-1]
    name = Path(s).name
    if name and len(name) < len(s):
        s = name
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def summarize_call(name, args):
    if not isinstance(args, dict):
        return name
    junk = {"null", "none", "undefined", "{}", "nil", "n/a", "()"}
    for key in ("path", "command", "pattern", "url", "source", "description", "workdir", "query"):
        val = args.get(key)
        if val is not None and str(val).strip() and str(val).strip().lower() not in junk:
            sval = _short_label(val, 48)
            return f"{name}  {sval}"
    return name
