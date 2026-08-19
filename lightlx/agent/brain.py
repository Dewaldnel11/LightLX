"""Cross-project LightLX brain: markdown files + idle extract jobs."""
import fnmatch
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..state import STATE_DIR
from .memory import MEMORY_ROOT, split_frontmatter

BRAIN_ROOT = Path(STATE_DIR) / "brain"
DIGEST_CAP_LINES = 120
DIGEST_CAP_BYTES = 16_000
KINDS = ("preference", "correction", "gotcha", "workflow", "project")
CONSOLIDATE_AFTER = 5

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_BEARER = re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret|password)\s*[:=]\s*\S+")
_SK = re.compile(r"\b(sk|ghp|gho|github_pat|hf)_[A-Za-z0-9._-]{8,}\b")
_AWS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PEM = re.compile(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.DOTALL)
_SSH = re.compile(r"(?i)(~/?\.ssh/[^\s]+)")


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ensure_layout(root=None) -> Path:
    root = Path(root or BRAIN_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("topics", "episodes", "imports"):
        (root / name).mkdir(exist_ok=True)
    for name in ("DIGEST.md", "corrections.md", "raw.md"):
        p = root / name
        if not p.exists():
            p.write_text("")
    return root


def format_record(kind, text, *, scope="global", source="user", confidence="high",
                  project="", session="", url=""):
    kind = (kind or "gotcha").lower().strip()
    if kind not in KINDS:
        kind = "gotcha"
    conf = (confidence or "high").lower().strip()
    if conf not in ("high", "medium", "low"):
        conf = "high"
    lines = [
        "---",
        f"kind: {kind}",
        f"scope: {scope or 'global'}",
        f"source: {source or 'user'}",
        f"confidence: {conf}",
        f"created: {_now_iso()}",
    ]
    if project:
        lines.append(f"project: {project}")
    if session:
        lines.append(f"session: {session}")
    if url:
        lines.append(f"url: {url}")
    lines += ["---", (text or "").strip(), ""]
    return "\n".join(lines)


def parse_record(text):
    return split_frontmatter(text or "")


def redact(text) -> str:
    if not text:
        return ""
    out = _PEM.sub("[redacted-pem]", text)
    out = _SK.sub("[redacted-token]", out)
    out = _AWS.sub("[redacted-key]", out)
    out = _BEARER.sub(lambda m: m.group(1) + "=[redacted]", out)
    out = _EMAIL.sub("[redacted-email]", out)
    out = _SSH.sub("[redacted-path]", out)
    return out


def _clip_digest(text) -> str:
    lines = (text or "").splitlines()
    clipped = "\n".join(lines[:DIGEST_CAP_LINES])
    if len(clipped.encode()) > DIGEST_CAP_BYTES:
        clipped = clipped.encode()[:DIGEST_CAP_BYTES].decode(errors="ignore")
    return clipped.strip()


def digest_for_prompt(root=None) -> str:
    root = Path(root or BRAIN_ROOT)
    p = root / "DIGEST.md"
    if not p.is_file():
        return ""
    body = _clip_digest(p.read_text(errors="replace"))
    if not body:
        return ""
    return "# Cross-project brain\n" + body


def _digest_line(kind, text, url=""):
    one = " ".join((text or "").strip().split())
    if len(one) > 160:
        one = one[:157] + "…"
    bit = f"- [{kind}] {one}"
    if url:
        bit += f" ({url})"
    return bit


def _should_digest(meta) -> bool:
    conf = str(meta.get("confidence") or "high").lower()
    source = str(meta.get("source") or "")
    url = str(meta.get("url") or "")
    if conf == "low":
        return False
    if source.startswith("idle-extract") and conf != "high":
        return False
    if source.startswith("idle-extract") and not url and conf != "high":
        return False
    return True


def write_record(kind, text, *, scope="global", source="user", confidence="high",
                 project="", session="", url="", root=None) -> str:
    root = ensure_layout(root)
    text = redact(text)
    rec = format_record(
        kind, text, scope=scope, source=source, confidence=confidence,
        project=project, session=session, url=url,
    )
    meta, body = parse_record(rec)
    kind = meta.get("kind") or kind
    if kind == "correction":
        with (root / "corrections.md").open("a") as f:
            f.write(rec if rec.endswith("\n") else rec + "\n")
    topic = re.sub(r"[^A-Za-z0-9._-]+", "-", (scope or kind).replace(":", "-"))[:40]
    topic_path = root / "topics" / f"{topic or kind}.md"
    with topic_path.open("a") as f:
        f.write(rec if rec.endswith("\n") else rec + "\n")
    if _should_digest(meta):
        line = _digest_line(kind, body, url or meta.get("url") or "")
        digest = root / "DIGEST.md"
        existing = digest.read_text(errors="replace") if digest.is_file() else ""
        if line not in existing:
            digest.write_text((existing.rstrip() + "\n" + line + "\n").lstrip())
            _clip_file(digest)
    return f"wrote brain record ({kind}, {meta.get('confidence')})"


def _clip_file(path: Path):
    text = _clip_digest(path.read_text(errors="replace"))
    path.write_text(text + ("\n" if text else ""))


def brain_search(query, *, kind="", scope="", max_hits=20, root=None) -> str:
    root = Path(root or BRAIN_ROOT)
    ensure_layout(root)
    max_hits = int(max_hits or 20)
    rx = re.compile(re.escape(query or ""), re.I)
    hits = []
    for base in (root, Path(MEMORY_ROOT)):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".md", ".txt"):
                continue
            try:
                blob = p.read_text(errors="replace")
            except Exception:
                continue
            if kind and f"kind: {kind}" not in blob and f"[{kind}]" not in blob:
                continue
            if scope and scope not in blob:
                continue
            for i, line in enumerate(blob.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p}:{i}:{line.strip()}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
    return "\n".join(hits) if hits else "(no matches)"


def rule_matches_paths(meta, touched_paths):
    paths = (meta or {}).get("paths")
    if not paths:
        return True
    if touched_paths is None:
        return False
    if isinstance(paths, str):
        paths = [paths]
    for tp in touched_paths:
        name = Path(str(tp)).name
        for g in paths:
            g = str(g)
            if fnmatch.fnmatch(str(tp), g) or fnmatch.fnmatch(name, g):
                return True
    return False


def import_instructions(lightlx_md=None) -> str:
    dest = Path(lightlx_md or Path(STATE_DIR) / "LIGHTLX.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.read_text(errors="replace") if dest.is_file() else ""
    added = []
    for label, src in (
        ("claude", Path.home() / ".claude" / "CLAUDE.md"),
        ("codex", Path.home() / ".codex" / "AGENTS.md"),
    ):
        if not src.is_file():
            continue
        ref = f"@{src}"
        if ref in existing or str(src) in existing:
            continue
        existing = (existing.rstrip() + f"\n{ref}\n").lstrip()
        added.append(label)
    if added:
        dest.write_text(existing)
        return "imported instruction refs: " + ", ".join(added)
    return "no new instruction files to import"


def import_foreign_memories(root=None) -> str:
    root = ensure_layout(root)
    notes = []
    mapping = {
        "codex": Path.home() / ".codex" / "memories",
        "claude": Path.home() / ".claude" / "memory",
    }
    for name, src in mapping.items():
        if not src.exists():
            continue
        dest = root / "imports" / name
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / "SOURCE.txt"
        marker.write_text(f"imported from {src}\n")
        if src.is_dir():
            for p in src.rglob("*.md"):
                target = dest / p.name
                if not target.exists():
                    try:
                        target.write_text(redact(p.read_text(errors="replace")))
                    except Exception:
                        continue
        notes.append(name)
    if not notes:
        return "no Claude/Codex memory dirs found"
    return "imported memories: " + ", ".join(notes) + " (DIGEST unchanged)"


def stage1_extract_prompt(history_text) -> str:
    return (
        "Extract lasting facts from this coding session. Return markdown records only, "
        "no preamble. Each record: YAML frontmatter with kind (preference|correction|gotcha|"
        "workflow|project), scope, source: idle-extract, confidence (high only if the USER "
        "stated it or a test proved it; otherwise low), optional url. Then the fact.\n"
        "Skip secrets, one-off task chatter, and anything you cannot ground in the transcript.\n\n"
        + (history_text or "")[:24_000]
    )


def stage1_save(session_id, extract_text, root=None) -> str:
    root = ensure_layout(root)
    text = redact((extract_text or "").strip())
    if not text or text in ("(empty)", "none", "no facts"):
        return "skipped empty extract"
    sid = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id or "session")[:80]
    (root / "episodes" / f"{sid}.md").write_text(text + "\n")
    with (root / "raw.md").open("a") as f:
        f.write(f"\n# {sid}\n{text}\n")
    return f"saved episode {sid}"


def history_text_from_record(rec) -> str:
    parts = []
    for m in rec.get("history") or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content") or ""
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        c = str(c).strip()
        if c:
            parts.append(f"{role}: {c}")
    return "\n".join(parts)


def should_extract_session(rec_or_path, *, idle_seconds, now=None, min_turns=4) -> bool:
    now = now if now is not None else time.time()
    path = None
    rec = rec_or_path
    if isinstance(rec_or_path, (str, Path)):
        path = Path(rec_or_path)
        try:
            rec = json.loads(path.read_text())
        except Exception:
            rec = {}
        mtime = path.stat().st_mtime
    else:
        rec = rec or {}
        mtime = now
        updated = rec.get("updated")
        if updated:
            try:
                mtime = datetime.fromisoformat(str(updated).replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
    if now - mtime < int(idle_seconds or 900):
        return False
    hist = rec.get("history") or []
    turns = sum(1 for m in hist if m.get("role") == "user")
    return turns >= int(min_turns or 4)


def _db(root=None):
    root = ensure_layout(root)
    p = root / "jobs.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs ("
        "session_id TEXT PRIMARY KEY, path TEXT, status TEXT, "
        "lease_until REAL, created REAL)"
    )
    conn.commit()
    return conn


def init_jobs(root=None):
    conn = _db(root)
    conn.close()


def enqueue(session_id, path, root=None):
    conn = _db(root)
    conn.execute(
        "INSERT OR IGNORE INTO jobs(session_id, path, status, lease_until, created) "
        "VALUES (?,?,?,?,?)",
        (session_id, str(path), "pending", 0, time.time()),
    )
    conn.commit()
    conn.close()


def claim(now=None, lease_seconds=600, root=None):
    now = now if now is not None else time.time()
    conn = _db(root)
    row = conn.execute(
        "SELECT session_id, path FROM jobs WHERE status='pending' "
        "OR (status='leased' AND lease_until < ?) LIMIT 1",
        (now,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    conn.execute(
        "UPDATE jobs SET status='leased', lease_until=? WHERE session_id=?",
        (now + int(lease_seconds or 600), row[0]),
    )
    conn.commit()
    conn.close()
    return {"session_id": row[0], "path": row[1]}


def complete(session_id, root=None):
    conn = _db(root)
    conn.execute("UPDATE jobs SET status='done' WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


def count_pending_episodes(root=None) -> int:
    root = Path(root or BRAIN_ROOT)
    d = root / "episodes"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.md"))


def consolidate(provider=None, root=None) -> str:
    root = ensure_layout(root)
    corr = root / "corrections.md"
    if corr.is_file():
        seen, keep = set(), []
        chunk = []
        for line in corr.read_text(errors="replace").splitlines(True):
            if line.startswith("---") and chunk:
                key = "".join(chunk)
                if key not in seen:
                    seen.add(key)
                    keep.append(key)
                chunk = [line]
            else:
                chunk.append(line)
        if chunk:
            key = "".join(chunk)
            if key not in seen:
                keep.append(key)
        corr.write_text("".join(keep))
    lines = []
    digest = root / "DIGEST.md"
    if digest.is_file():
        for line in digest.read_text(errors="replace").splitlines():
            if line.strip() and line not in lines:
                lines.append(line)
    digest.write_text("\n".join(lines[:DIGEST_CAP_LINES]) + ("\n" if lines else ""))
    _clip_file(digest)
    return "consolidated brain"


def tick_idle(prefs, *, busy=False, provider=None, list_session_files=None, root=None):
    prefs = prefs or {}
    enabled = prefs.get("brain.enabled", prefs.get("brain_enabled", False))
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "on", "yes")
    if not enabled or busy:
        return ""
    idle = int(prefs.get("brain.idle_seconds") or prefs.get("brain_idle_seconds") or 900)
    files = list(list_session_files() or []) if list_session_files else []
    for p in files:
        p = Path(p)
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if should_extract_session(p, idle_seconds=idle):
            enqueue(rec.get("id") or p.stem, p, root=root)
    job = claim(root=root)
    if not job:
        return ""
    extract = ""
    try:
        rec = json.loads(Path(job["path"]).read_text())
        hist = history_text_from_record(rec)
        if provider is not None and hist:
            from .types import Completion
            prompt = stage1_extract_prompt(hist)
            comp = provider.complete(
                [{"role": "user", "content": prompt}],
                tools=None, max_tokens=800, temperature=0.1,
            )
            extract = getattr(comp, "content", None) or (comp if isinstance(comp, str) else "")
            if isinstance(comp, Completion):
                extract = comp.content or ""
    except Exception:
        extract = ""
    msg = stage1_save(job["session_id"], extract, root=root)
    complete(job["session_id"], root=root)
    if count_pending_episodes(root) >= CONSOLIDATE_AFTER:
        consolidate(provider=None, root=root)
    return msg


def parse_ddg_html(html) -> list:
    """Pull title+url pairs from DuckDuckGo HTML results."""
    out = []
    for m in re.finditer(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html or "", re.I | re.DOTALL,
    ):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        href = unquote(href.replace("&amp;", "&"))
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            href = unquote((qs.get("uddg") or [href])[0])
        title = re.sub(r"\s+", " ", title).strip()
        if href.startswith("http") and title:
            out.append({"title": title, "url": href})
    if not out:
        for m in re.finditer(r'href="(https?://[^"]+)"[^>]*>([^<]{8,120})', html or ""):
            href, title = m.group(1), m.group(2).strip()
            if "duckduckgo.com" in href:
                continue
            out.append({"title": title, "url": href})
            if len(out) >= 8:
                break
    seen, uniq = set(), []
    for item in out:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        uniq.append(item)
    return uniq[:10]


def claims_missing_sources(plan_text) -> list:
    """Lines that look like factual claims without a URL — local models must not treat these as verified."""
    missing = []
    for line in (plan_text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        if s.startswith(("- ", "* ", "1.", "2.", "3.")) or "|" in s:
            if "http://" not in s and "https://" not in s and "source:" not in s.lower():
                if len(s) > 24:
                    missing.append(s)
    return missing
