import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..state import STATE_DIR
from .memory import git_root, split_frontmatter

BANG = re.compile(r"^!`([^`]+)`\s*$", re.MULTILINE)
BANG_FENCE = re.compile(r"^```!\s*\n(.*?)```", re.DOTALL | re.MULTILINE)


BUNDLED = {
    "code-review": {
        "description": "Review uncommitted changes for bugs, regressions, and missing tests. Use when the user asks for a review or what looks risky in the diff.",
        "body": """## Current diff

!`git diff HEAD`

## Instructions

Review the diff. Report:
- bugs and edge cases
- missing tests
- API / behaviour regressions
- anything unsafe

Be specific (file + why). If the diff is empty, say so.
""",
    },
    "summarize-changes": {
        "description": "Summarize uncommitted work and suggest a commit message. Use when the user asks what changed or wants a commit message.",
        "body": """## Current changes

!`git status -sb`
!`git diff HEAD`

## Instructions

Summarize in 2–4 bullets, then suggest a conventional commit subject line. If nothing changed, say so.
""",
    },
}


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str
    path: Path | None = None
    disable_model: bool = False
    user_invocable: bool = True
    when_to_use: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def fork(self) -> bool:
        return str(self.extras.get("context") or "").lower() == "fork"

    @property
    def agent_kind(self) -> str:
        kind = str(self.extras.get("agent") or "general").lower()
        return kind if kind in ("explore", "implement", "general") else "general"

    def listing(self) -> str:
        desc = self.description
        if self.when_to_use:
            desc = f"{desc} {self.when_to_use}"
        if len(desc) > 240:
            desc = desc[:237] + "…"
        return f"- {self.name}: {desc}"


HOME_SKILL_DIRS = (
    ("claude-user", Path.home() / ".claude" / "skills"),
    ("claude-user-commands", Path.home() / ".claude" / "commands"),
    ("codex-user", Path.home() / ".codex" / "skills"),
    ("codex-user-prompts", Path.home() / ".codex" / "prompts"),
    ("agents-user", Path.home() / ".agents" / "skills"),
)


def _skill_dirs(workspace: Path) -> list[tuple[str, Path]]:
    ws = Path(workspace).resolve()
    root = git_root(ws) or ws
    out = list(HOME_SKILL_DIRS)
    out.append(("user", Path(STATE_DIR) / "skills"))
    chain = []
    cur = ws
    while True:
        chain.append(cur)
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    for folder in reversed(chain):
        out.append(("project", folder / ".lightlx" / "skills"))
        out.append(("claude", folder / ".claude" / "skills"))
        out.append(("codex", folder / ".codex" / "skills"))
        out.append(("agents", folder / ".agents" / "skills"))
        cmd = folder / ".lightlx" / "commands"
        out.append(("commands", cmd))
        out.append(("claude-commands", folder / ".claude" / "commands"))
    return out


def _load_skill_md(name, path: Path, source: str) -> Skill | None:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None
    meta, body = split_frontmatter(text)
    desc = str(meta.get("description") or "").strip()
    if not desc:
        desc = next((ln.strip() for ln in body.splitlines() if ln.strip()), name)
    flag = str(meta.get("disable-model-invocation") or meta.get("disable_model_invocation") or "").lower()
    inv = str(meta.get("user-invocable") or meta.get("user_invocable") or "true").lower()
    return Skill(
        name=str(meta.get("name") or name),
        description=desc,
        body=body.strip(),
        source=source,
        path=path,
        disable_model=flag in ("true", "yes", "1", "on"),
        user_invocable=inv not in ("false", "no", "0", "off"),
        when_to_use=str(meta.get("when_to_use") or meta.get("when-to-use") or ""),
        extras=meta,
    )


def discover_skills(workspace) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for name, spec in BUNDLED.items():
        found[name] = Skill(name, spec["description"], spec["body"], "bundled")
    for source, folder in _skill_dirs(Path(workspace)):
        if not folder.is_dir():
            continue
        if folder.name in ("commands", "prompts"):
            for p in sorted(folder.glob("*.md")):
                sk = _load_skill_md(p.stem, p, source)
                if sk:
                    found[sk.name] = sk
            continue
        for child in sorted(folder.iterdir()):
            md = child / "SKILL.md" if child.is_dir() else None
            if md and md.is_file():
                sk = _load_skill_md(child.name, md, source)
                if sk:
                    found[sk.name] = sk
    return found


def expand_skill(skill: Skill, arguments="", workspace=".") -> str:
    text = skill.body or ""
    args = arguments or ""
    parts = args.split()
    text = text.replace("$ARGUMENTS", args)
    text = text.replace("${ARGUMENTS}", args)
    for i, p in enumerate(parts):
        text = text.replace(f"$ARGUMENTS[{i}]", p)
        text = text.replace(f"${i}", p)

    def run_cmd(cmd):
        try:
            r = subprocess.run(
                cmd, shell=True, cwd=str(workspace), capture_output=True,
                text=True, timeout=30, env=os.environ.copy(),
            )
            out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
            return out.strip() or f"(no output, exit {r.returncode})"
        except Exception as e:
            return f"(command failed: {e})"

    def bang(m):
        return run_cmd(m.group(1))

    def fence(m):
        return run_cmd(m.group(1).strip())

    text = BANG.sub(bang, text)
    text = BANG_FENCE.sub(fence, text)
    extra = ""
    if skill.path and skill.path.parent.is_dir() and skill.path.name == "SKILL.md":
        sibs = [p.name for p in skill.path.parent.iterdir() if p.name != "SKILL.md" and not p.name.startswith(".")]
        if sibs:
            extra = "\n\nSupporting files in " + str(skill.path.parent) + ": " + ", ".join(sibs)
    return f"# Skill: {skill.name}\n\n{text}{extra}".strip()


def catalog(skills: dict[str, Skill]) -> str:
    auto = [s for s in skills.values() if not s.disable_model]
    manual = [s for s in skills.values() if s.disable_model and s.user_invocable]
    if not auto and not manual:
        return ""
    lines = ["# Skills", "Load a skill with the skill tool when it matches the task. Users can also type /name."]
    if auto:
        lines.append("Available (you may load these):")
        lines.extend(s.listing() for s in sorted(auto, key=lambda x: x.name))
    if manual:
        lines.append("User-only (do not auto-load):")
        lines.extend(s.listing() for s in sorted(manual, key=lambda x: x.name))
    return "\n".join(lines)


def list_importable(workspace=None) -> list[Skill]:
    found = []
    seen = set()
    dirs = list(HOME_SKILL_DIRS)
    if workspace:
        ws = Path(workspace).resolve()
        root = git_root(ws) or ws
        for folder in (root, ws):
            dirs.extend([
                ("claude", folder / ".claude" / "skills"),
                ("codex", folder / ".codex" / "skills"),
                ("claude-commands", folder / ".claude" / "commands"),
            ])
    for source, folder in dirs:
        if not folder.is_dir():
            continue
        if folder.name in ("commands", "prompts"):
            kids = [(p.stem, p) for p in folder.glob("*.md")]
        else:
            kids = [(c.name, c / "SKILL.md") for c in folder.iterdir() if c.is_dir()]
        for name, path in kids:
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            sk = _load_skill_md(name, path, source)
            if sk:
                found.append(sk)
    return found


def import_skills(skills: list[Skill], dest: Path, link=False) -> list[str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    reports = []
    for sk in skills:
        if not sk.path:
            reports.append(f"skip {sk.name} (no file)")
            continue
        src = sk.path.parent if sk.path.name == "SKILL.md" else sk.path
        target = dest / sk.name
        if target.exists() or target.is_symlink():
            reports.append(f"exists {sk.name}")
            continue
        try:
            if link:
                os.symlink(src, target, target_is_directory=src.is_dir())
                reports.append(f"linked {sk.name} → {src}")
            elif src.is_dir():
                import shutil
                shutil.copytree(src, target)
                reports.append(f"copied {sk.name}/")
            else:
                target.mkdir()
                import shutil
                shutil.copy2(src, target / "SKILL.md")
                reports.append(f"copied {sk.name}")
        except Exception as e:
            reports.append(f"fail {sk.name}: {e}")
    return reports
