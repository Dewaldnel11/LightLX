import hashlib
import os
import re
import subprocess
from pathlib import Path

from ..state import STATE_DIR

INSTR_NAMES = (
    "LIGHTLX.md",
    "LIGHTLX.local.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
)
NESTED = (
    Path(".lightlx") / "LIGHTLX.md",
    Path(".claude") / "CLAUDE.md",
    Path(".codex") / "AGENTS.md",
)
RULE_DIRS = (Path(".lightlx") / "rules", Path(".claude") / "rules", Path(".codex") / "rules")
USER_INSTR = Path(STATE_DIR) / "LIGHTLX.md"
USER_RULES = Path(STATE_DIR) / "rules"
MEMORY_ROOT = Path(STATE_DIR) / "memory"
INDEX_CAP_LINES = 200
INDEX_CAP_BYTES = 25_000
TOTAL_CAP = 80_000
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_IMPORT = re.compile(r"(?<!`)@([^\s`]+)")


def git_root(start: Path) -> Path | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start), capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return None


def project_slug(workspace: Path) -> str:
    root = git_root(workspace) or Path(workspace).resolve()
    raw = str(root).replace(os.path.expanduser("~"), "~")
    h = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")[:60]
    return f"{slug}-{h}"


def split_frontmatter(text: str):
    text = text or ""
    if not text.startswith("---"):
        return {}, text
    nl = text.find("\n")
    if nl < 0:
        return {}, text
    end = text.find("\n---", nl)
    if end < 0:
        return {}, text
    raw = text[nl + 1:end]
    body = text[end + 4:].lstrip("\n")
    meta = {}
    key = None
    for line in raw.splitlines():
        if re.match(r"^\s+-\s+", line) and key:
            meta.setdefault(key, [])
            if not isinstance(meta[key], list):
                meta[key] = [meta[key]]
            meta[key].append(line.split("-", 1)[1].strip().strip("\"'"))
            continue
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            key = k.strip()
            v = v.strip().strip("\"'")
            meta[key] = [] if v == "" else v
    return meta, body


def _read(path: Path, cap=TOTAL_CAP) -> str:
    try:
        data = path.read_text(errors="replace")
    except Exception:
        return ""
    data = _COMMENT.sub("", data)
    if len(data) > cap:
        data = data[:cap] + "\n… truncated"
    return data


def _expand_imports(text: str, base: Path, depth=0, seen=None) -> str:
    if depth > 4:
        return text
    seen = seen if seen is not None else set()

    def repl(m):
        ref = m.group(1).rstrip(".,;:)")
        p = Path(os.path.expanduser(ref))
        if not p.is_absolute():
            p = (base / ref).resolve()
        key = str(p)
        if key in seen or not p.is_file():
            return m.group(0)
        seen.add(key)
        return _expand_imports(_read(p, 30_000), p.parent, depth + 1, seen)

    parts, last = [], 0
    in_fence = False
    for line in text.splitlines(True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            parts.append(line)
            continue
        if in_fence or "`" in line:
            parts.append(line)
            continue
        parts.append(_IMPORT.sub(repl, line))
        last += 1
    return "".join(parts)


def _walk_dirs(workspace: Path):
    ws = Path(workspace).resolve()
    root = git_root(ws) or ws
    chain = []
    cur = ws
    while True:
        chain.append(cur)
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    chain.reverse()
    return chain


def load_instructions(workspace) -> list[tuple[str, str]]:
    ws = Path(workspace).resolve()
    out = []
    seen = set()

    def add(label, path: Path):
        key = str(path.resolve()) if path.exists() else None
        if not key or key in seen:
            return
        seen.add(key)
        body = _expand_imports(_read(path), path.parent)
        if body.strip():
            out.append((label, body.strip()))

    if USER_INSTR.is_file():
        add("user LIGHTLX.md", USER_INSTR)
    if USER_RULES.is_dir():
        for p in sorted(USER_RULES.rglob("*.md")):
            add(f"user rules/{p.name}", p)

    for folder in _walk_dirs(ws):
        for name in INSTR_NAMES:
            add(f"{folder.name}/{name}", folder / name)
        for rel in NESTED:
            add(str(rel), folder / rel)
        for rd in RULE_DIRS:
            d = folder / rd
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*.md")):
                meta, body = split_frontmatter(_read(p))
                paths = meta.get("paths")
                if paths:
                    continue
                if body.strip():
                    add(f"rules/{p.name}", p)

    return out


def format_instructions(blocks: list[tuple[str, str]], limit=TOTAL_CAP) -> str:
    if not blocks:
        return ""
    parts = ["# Project instructions (always follow)"]
    used = 0
    for label, body in blocks:
        chunk = f"\n## {label}\n{body}\n"
        if used + len(chunk) > limit:
            parts.append("\n… further instruction files omitted")
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts).strip()


class MemoryStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.dir = MEMORY_ROOT / project_slug(self.workspace)
        self.index = self.dir / "MEMORY.md"

    def ensure(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.index.exists():
            self.index.write_text(
                "# Memory index\n\nOne line per fact. Detail goes in topic files in this folder.\n"
            )

    def load_index(self) -> str:
        if not self.index.is_file():
            return ""
        text = _COMMENT.sub("", self.index.read_text(errors="replace"))
        lines = text.splitlines()
        clipped = "\n".join(lines[:INDEX_CAP_LINES])
        if len(clipped.encode()) > INDEX_CAP_BYTES:
            clipped = clipped.encode()[:INDEX_CAP_BYTES].decode(errors="ignore")
        return clipped.strip()

    def list_files(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.name for p in self.dir.glob("*.md"))

    def read(self, name="MEMORY.md") -> str:
        name = Path(name).name
        p = self.dir / name
        if not p.is_file():
            return f"error: no memory file {name}"
        return _read(p, 60_000)

    def write(self, name, content) -> str:
        self.ensure()
        name = Path(name or "MEMORY.md").name
        if not name.endswith(".md"):
            name += ".md"
        p = self.dir / name
        p.write_text(content if isinstance(content, str) else str(content))
        n = len(content.splitlines()) if isinstance(content, str) else 0
        note = ""
        if name == "MEMORY.md" and (n > INDEX_CAP_LINES or len(content) > INDEX_CAP_BYTES):
            note = " — index is over the load limit; keep one line per entry and move detail to topic files"
        return f"wrote memory/{name} ({n} lines){note}"

    def format_for_prompt(self) -> str:
        idx = self.load_index()
        if not idx:
            return ""
        extras = [n for n in self.list_files() if n != "MEMORY.md"]
        tail = f"\nTopic files (read with memory): {', '.join(extras)}" if extras else ""
        return "# Auto memory (your notes from prior sessions)\n" + idx + tail


def init_project(workspace) -> str:
    ws = Path(workspace).resolve()
    target = ws / "LIGHTLX.md"
    existed = target.is_file()
    bits = ["# LightLX", "", "Instructions for the coding agent in this repo.", ""]
    readme = ws / "README.md"
    if readme.is_file():
        first = next((ln[2:].strip() for ln in readme.read_text(errors="replace").splitlines() if ln.startswith("# ")), "")
        if first:
            bits += [f"Project: {first}", ""]
    cmds = []
    if (ws / "pyproject.toml").is_file() or (ws / "setup.py").is_file():
        cmds += ["- Python: `python -m unittest` or the project's test extra"]
    if (ws / "package.json").is_file():
        cmds += ["- Node: see `package.json` scripts (`npm test`, `npm run lint`)"]
    if (ws / "Makefile").is_file():
        cmds += ["- Make: see `Makefile`"]
    if (ws / "Cargo.toml").is_file():
        cmds += ["- Rust: `cargo test`"]
    if cmds:
        bits += ["## Checks", *cmds, ""]
    bits += [
        "## Conventions",
        "- Read before you edit. Keep diffs small.",
        "- Do not add comments unless asked.",
        "- After changes, run the relevant check above.",
        "",
        "## Memory",
        "Save lasting facts with the memory tool. Do not put secrets in LIGHTLX.md.",
        "",
    ]
    if existed:
        return f"LIGHTLX.md already exists at {target} — not overwritten. Edit it, or add skills under .lightlx/skills/"
    target.write_text("\n".join(bits))
    local = ws / "LIGHTLX.local.md"
    hint = ""
    if not local.exists():
        hint = f" Optional personal notes: create {local.name} (gitignored)."
    return f"wrote {target}{hint}"
