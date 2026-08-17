import os
import sys
import time

from .context import detect_context, estimate_tokens, handoff_note, maybe_compact
from .loop import SubagentLauncher, run_agent, seed_messages
from .mcp import MCPHub, load_mcp_config
from .memory import format_instructions, init_project, load_instructions
from .memory import MemoryStore
from .prompts import DOC_ALIASES
from .sessions import age, hydrate_messages, list_sessions, load_session, record_from, save_session, title_from
from .skills import catalog, discover_skills, expand_skill, import_skills, list_importable
from .tools import BuiltinTools, Workspace

HELP = """commands
  /menu          settings
  /tools         list tools (builtin + MCP)
  /skills        list skills (LIGHTLX / Claude / Codex)
  /import        import Claude / Codex skills into LightLX
  /memory        show auto-memory files
  /init          write a starter LIGHTLX.md for this repo
  /mcp           MCP servers — status / reload
  /docs          documentation sources (claude-code, codex, …)
  /workspace     show or set the working directory
  /task          reminder: ask the model to spawn parallel subagents
  /tokens N      max tokens per model call
  /compact       summarize older turns to free context
  /resume        resume a saved session
  /save          snapshot this session now
  /handoff       switch model and keep this conversation
  /clear         forget the conversation
  /model         switch model / backend (keeps chat — same as /handoff)
  /help          show this
  /exit          quit
type /skill-name to run a skill   ·   Ctrl-C stops a turn"""


def _dim(s):
    return f"\033[2m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def _ok(s):
    return f"\033[32m{s}\033[0m"


def _err(s):
    return f"\033[31m{s}\033[0m"


class AgentSession:
    def __init__(self, provider, workspace, prefs, on_switch=None, native_tools=True, source=None):
        self.provider = provider
        self.ws = Workspace(workspace)
        self.prefs = prefs
        self.on_switch = on_switch
        self.native_tools = native_tools
        self.source = source or {}
        self.history = []
        self.pending = None
        self.session_id = None
        self.hub = MCPHub()
        self.max_tokens = int(prefs.get("agent_max_tokens") or prefs.get("max_tokens") or 4096)
        self.temperature = float(prefs.get("temperature") or 0.1)
        self.context_length = detect_context(provider)
        self.memory = MemoryStore(self.ws.root)
        self.skills = discover_skills(self.ws.root)
        self.instructions = load_instructions(self.ws.root)
        self._connect_mcp()

    def reload_knowledge(self):
        self.memory = MemoryStore(self.ws.root)
        self.skills = discover_skills(self.ws.root)
        self.instructions = load_instructions(self.ws.root)

    def prompt_extra(self) -> str:
        parts = [
            format_instructions(self.instructions),
            catalog(self.skills),
            self.memory.format_for_prompt(),
        ]
        return "\n\n".join(p for p in parts if p)

    def persist(self):
        if not self.history and not self.pending:
            return
        rec = record_from(self, self.source)
        self.session_id = save_session(rec)

    def checkpoint(self, messages):
        self.pending = [m for m in (messages or []) if m.get("role") != "system"]
        self.persist()

    def apply_handoff(self, old_label):
        new = getattr(self.provider, "label", "")
        self.history, did = maybe_compact(
            self.provider, self.history, self.context_length, self.max_tokens,
        )
        self.history.append(handoff_note(old_label, new, self.context_length))
        return did

    def _connect_mcp(self):
        self.hub.close()
        cfg = load_mcp_config(str(self.ws.root))
        if not cfg:
            return
        self.hub.connect(cfg, str(self.ws.root), on_status=None)

    def make_registry(self, kind="general", depth=0):
        include_task = depth < 1
        readonly = kind == "explore"
        if kind == "explore":
            include_task = False
        launcher = None
        if include_task:
            launcher = SubagentLauncher(
                self.provider, lambda kind="general", depth=1: self.make_registry(kind, depth),
                str(self.ws.root), self.max_tokens, self.temperature,
                on_event=None, native_tools=self.native_tools, extra=self.prompt_extra(),
            )
        builtin = BuiltinTools(self.ws, launch_task=launcher, skills=self.skills, memory=self.memory)
        specs = builtin.specs(include_task=include_task, readonly=readonly)
        if kind != "explore":
            specs.extend(self.hub.tool_specs())
        return {s.name: s for s in specs}

    def close(self):
        self.hub.close()


def _indent(depth):
    return "  " + ("  " * depth)


def make_printer():
    started = {"n": False}
    tool_stack = {}
    read_group = {"active": False, "names": []}
    subagent_stack = {}

    def on_event(kind, text="", name="", detail="", depth=0, ok=True, **_):
        pad = "  " + ("  " * depth)
        if kind == "text":
            if not started["n"]:
                print()
                started["n"] = True
            sys.stdout.write(text)
            sys.stdout.flush()
        elif kind == "tool_start":
            if started["n"]:
                print()
                started["n"] = False
            tool_stack[name] = {"detail": detail or name, "t0": time.time()}
            if name in ("read_file", "list_dir", "glob", "grep", "fetch_url", "docs"):
                read_group["active"] = True
                read_group["names"].append(detail or name)
                _render_read_group(pad)
            else:
                print(_dim(f"{pad}▸ {detail or name}"))
        elif kind == "tool_end":
            if started["n"]:
                print()
                started["n"] = False
            info = tool_stack.pop(name, {})
            dur = time.time() - info.get("t0", time.time())
            mark = _ok("✓") if ok else _err("✗")
            tail = f" {_dim(f'{dur:.1f}s')}" if dur > 0.2 else ""
            if name in ("read_file", "list_dir", "glob", "grep", "fetch_url", "docs"):
                read_group["names"] = [n for n in read_group["names"] if n != (detail or name)]
                if not read_group["names"]:
                    read_group["active"] = False
                    sys.stdout.write("\033[K")
                    sys.stdout.flush()
                else:
                    _render_read_group(pad)
            else:
                print(_dim(f"{pad}{mark} {info.get('detail', name)}{tail}"))
        elif kind == "subagent_start":
            if started["n"]:
                print()
                started["n"] = False
            subagent_stack[name] = detail
            # don't print start - wait for end to show one compact line
        elif kind == "subagent_end":
            if started["n"]:
                print()
                started["n"] = False
            detail = subagent_stack.pop(name, detail)
            mark = _ok("✓") if ok else _err("✗")
            # one compact line for the whole subagent
            short = detail[:60] + ("…" if len(detail) > 60 else "")
            print(_dim(f"{pad}{mark} subagent: {short}"))
        elif kind == "compact":
            if started["n"]:
                print()
                started["n"] = False
            print(_dim(f"{pad}◎ compacting · {detail}"))
        elif kind == "compact_done":
            print(_dim(f"{pad}◎ compacted"))
        elif kind == "error":
            print(_err(f"\n{pad}{text}"))
        elif kind == "turn_end":
            if started["n"]:
                print()
                started["n"] = False

    def _render_read_group(pad):
        names = read_group["names"]
        if len(names) == 1:
            label = names[0]
        elif len(names) <= 3:
            label = ", ".join(names)
        else:
            label = f"{', '.join(names[:2])} +{len(names)-2} more"
        line = f"{pad}▸ reading {label}"
        sys.stdout.write("\033[K" + line + "\033[" + str(len(line)) + "D")
        sys.stdout.flush()

    def reset():
        started["n"] = False
        tool_stack.clear()
        read_group["active"] = False
        read_group["names"] = []
        subagent_stack.clear()

    on_event.reset = reset
    return on_event


def _bind_events(sess, on_event):
    def make_registry(kind="general", depth=0):
        include_task = depth < 1 and kind != "explore"
        readonly = kind == "explore"
        launcher = None
        if include_task:
            launcher = SubagentLauncher(
                sess.provider, make_registry, str(sess.ws.root),
                sess.max_tokens, sess.temperature, on_event, sess.native_tools,
                extra=sess.prompt_extra(),
            )
        builtin = BuiltinTools(sess.ws, launch_task=launcher, skills=sess.skills, memory=sess.memory)
        specs = builtin.specs(include_task=include_task, readonly=readonly)
        if kind != "explore":
            specs.extend(sess.hub.tool_specs())
        return {s.name: s for s in specs}

    sess.make_registry = make_registry
    return make_registry


def _finish_turn(sess, result):
    sess.pending = None
    sess.history = [m for m in (result.messages or []) if m.get("role") != "system"]
    sess.persist()


def resume_pending(sess, on_event):
    registry = _bind_events(sess, on_event)
    rest = hydrate_messages(sess.pending)
    messages = seed_messages(
        str(sess.ws.root), registry(kind="general", depth=0),
        rest, tools_as_text=not sess.native_tools,
        extra=sess.prompt_extra(),
    )
    print(_dim("  resuming in-flight turn…"))
    try:
        result = run_agent(
            sess.provider, messages, registry(kind="general", depth=0),
            max_tokens=sess.max_tokens, temperature=sess.temperature,
            on_event=on_event, native_tools=sess.native_tools,
            context_length=sess.context_length,
            on_checkpoint=sess.checkpoint,
        )
    except KeyboardInterrupt:
        print(_dim("\n— stopped — checkpoint saved. resume next launch."))
        return
    except Exception as e:
        print(_dim(f"\n  {e}"))
        return
    _finish_turn(sess, result)


def run_turn(sess, user_text, on_event):
    registry = _bind_events(sess, on_event)
    sess.history.append({"role": "user", "content": user_text})
    sess.history, _ = maybe_compact(
        sess.provider, sess.history, sess.context_length, sess.max_tokens, on_event=on_event,
    )
    messages = seed_messages(
        str(sess.ws.root), registry(kind="general", depth=0),
        sess.history, tools_as_text=not sess.native_tools,
        extra=sess.prompt_extra(),
    )
    sess.checkpoint(messages)
    try:
        result = run_agent(
            sess.provider, messages, registry(kind="general", depth=0),
            max_tokens=sess.max_tokens, temperature=sess.temperature,
            on_event=on_event, native_tools=sess.native_tools,
            context_length=sess.context_length,
            on_checkpoint=sess.checkpoint,
        )
    except KeyboardInterrupt:
        print(_dim("\n— stopped — checkpoint saved. pick Resume next time."))
        return
    except Exception as e:
        print(_dim(f"\n  {e}"))
        sess.history.pop()
        sess.pending = None
        sess.persist()
        return
    _finish_turn(sess, result)


def _prompt_line(sess):
    used = estimate_tokens(sess.history)
    ctx = sess.context_length or 0
    ctx_s = f"{used}/{ctx}" if ctx else str(used)
    return "\n" + _dim(f"{ctx_s}") + "  " + _bold(sess.provider.label) + " " + _bold("›") + " "


def settings_menu(sess):
    while True:
        print("\n  " + _bold("Settings") + _dim("   number · Enter to back"))
        print(f"   1  Reply length   {_bold(str(sess.max_tokens))}")
        print(f"   2  Workspace      {_dim(str(sess.ws.root))}")
        print(f"   3  Handoff model")
        print(f"   4  Reload MCP     {_dim(str(len(sess.hub.servers)) + ' connected')}")
        print(f"   5  Compact now    {_dim(str(estimate_tokens(sess.history)) + ' / ' + str(sess.context_length))}")
        print("   q  Quit")
        try:
            c = input("  " + _bold("›") + " ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ""
        if c in ("", "b", "back"):
            return ""
        if c == "1":
            try:
                val = input(_dim(f"  tokens (now {sess.max_tokens}) › ")).strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if val.isdigit() and int(val) > 0:
                sess.max_tokens = sess.prefs["agent_max_tokens"] = int(val)
                print(_dim(f"  reply length = {sess.max_tokens}"))
        elif c == "2":
            _set_workspace(sess)
        elif c == "3":
            return "switch"
        elif c == "4":
            sess._connect_mcp()
            print(_dim("  MCP reloaded"))
        elif c == "5":
            _do_compact(sess)
        elif c in ("q", "quit"):
            return "quit"
        else:
            print(_dim("  pick a number"))


def _set_workspace(sess, arg=None):
    val = arg
    if val is None:
        try:
            val = input(_dim(f"  workspace (now {sess.ws.root}) › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
    if not val:
        print(_dim(f"  {sess.ws.root}"))
        return
    path = os.path.abspath(os.path.expanduser(val))
    if not os.path.isdir(path):
        print(_dim(f"  not a directory: {path}"))
        return
    sess.ws = Workspace(path)
    os.chdir(path)
    sess.prefs["workspace"] = path
    sess.reload_knowledge()
    sess._connect_mcp()
    print(_dim(f"  workspace = {path}"))


def _do_compact(sess):
    before = estimate_tokens(sess.history)
    sess.history, did = maybe_compact(
        sess.provider, sess.history, sess.context_length, sess.max_tokens, force=True,
    )
    after = estimate_tokens(sess.history)
    if did:
        sess.persist()
        print(_dim(f"  compacted {before} → {after} tok"))
    else:
        print(_dim("  nothing to compact"))


def _pick_resume():
    rows = list_sessions(15)
    if not rows:
        print(_dim("  no saved sessions"))
        return None
    print("\n  " + _bold("Sessions"))
    for i, s in enumerate(rows, 1):
        label = s.get("title") or "untitled"
        if s.get("in_progress") or s.get("pending"):
            label = "● " + label
        meta = f"{s.get('provider') or s.get('kind') or '?'} · {age(s.get('updated'))}"
        print(f"   {i}  {label:<40} {_dim(meta)}")
    try:
        raw = input("  " + _bold("›") + " ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw or raw.lower() in ("q", "back"):
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1]
    hit = load_session(raw)
    return hit


def _import_menu(sess):
    from pathlib import Path
    found = list_importable(sess.ws.root)
    if not found:
        print(_dim("  nothing to import. looked in ~/.claude/skills, ~/.codex/skills, and this repo"))
        return
    print("\n  " + _bold("Import Claude / Codex skills"))
    for i, s in enumerate(found, 1):
        print(f"   {i}  {s.name:<22} {_dim(s.source)}  {s.description[:56]}")
    print(f"   a  all of them")
    print(f"   l  symlink all (stay in sync with Claude/Codex)")
    print(_dim("   Enter to back"))
    try:
        raw = input("  " + _bold("›") + " ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return
    link = raw == "l"
    if raw in ("a", "l"):
        chosen = found
    elif raw.isdigit() and 1 <= int(raw) <= len(found):
        chosen = [found[int(raw) - 1]]
    else:
        print(_dim("  pick a number, a, or l"))
        return
    print("   1  user  ~/.lightlx/skills   (all projects)")
    print("   2  this repo  .lightlx/skills")
    try:
        where = input("  " + _bold("›") + " ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    from ..state import STATE_DIR
    dest = Path(STATE_DIR) / "skills" if where != "2" else Path(sess.ws.root) / ".lightlx" / "skills"
    for line in import_skills(chosen, dest, link=link):
        print(_dim("  " + line))
    sess.reload_knowledge()
    print(_dim(f"  {len(sess.skills)} skills loaded"))


def apply_record(sess, rec):
    if not rec:
        return False
    sess.history = list(rec.get("history") or [])
    sess.pending = rec.get("pending") or None
    sess.session_id = rec.get("id")
    ws = rec.get("workspace")
    if ws and os.path.isdir(ws):
        sess.ws = Workspace(ws)
    print(_dim(f"  resumed {rec.get('title') or rec.get('id')} · {len(sess.history)} turns"))
    return True


def agent_repl(sess, switch_cb=None):
    ctx = sess.context_length
    print(_dim(f"\n  {sess.provider.label}  ·  ctx {ctx}"))
    print(_dim(f"  {sess.ws.root}"))
    print(_dim("  /menu · /skills · /memory · /help · /exit"))
    printer = make_printer()
    if sess.pending:
        resume_pending(sess, printer)
    while True:
        try:
            line = input(_prompt_line(sess)).strip()
        except (EOFError, KeyboardInterrupt):
            sess.persist()
            return
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit", "/q"):
            sess.persist()
            return
        if line in ("/menu", "/", "/settings"):
            action = settings_menu(sess)
            if action == "quit":
                sess.persist()
                return
            if action == "switch":
                sess.persist()
                return "switch"
            continue
        if line == "/help":
            print(HELP)
            continue
        if line in ("/clear", "/reset", "/new"):
            sess.persist()
            sess.history = []
            sess.pending = None
            sess.session_id = None
            print(_dim("  conversation cleared — new session"))
            continue
        if line == "/tools":
            for spec in sess.make_registry().values():
                src = f"  {_dim(spec.source)}" if spec.source != "builtin" else ""
                print(f"  {spec.name}{src}")
            continue
        if line == "/docs":
            print(_dim("  aliases → GitHub repos (tool: docs)"))
            seen = {}
            for k, v in DOC_ALIASES.items():
                seen.setdefault(v, []).append(k)
            for repo, aliases in seen.items():
                print(f"  {', '.join(aliases):<28} {repo}")
            print(_dim("  or pass any owner/repo"))
            continue
        if line == "/mcp":
            if sess.hub.servers:
                for line_s in sess.hub.summary():
                    print("  " + line_s)
            else:
                print(_dim("  no MCP servers. add ~/.lightlx/mcp.json  (see /help)"))
            continue
        if line == "/task":
            print(_dim("  ask the model to use the task tool — several in one turn run in parallel"))
            print(_dim("  types: explore (read-only) · implement · general"))
            continue
        if line.startswith("/workspace"):
            parts = line.split(maxsplit=1)
            _set_workspace(sess, parts[1] if len(parts) == 2 else None)
            continue
        if line.startswith("/tokens"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                sess.max_tokens = sess.prefs["agent_max_tokens"] = int(parts[1])
                print(_dim(f"  reply length = {sess.max_tokens}"))
            else:
                print(_dim(f"  {sess.max_tokens}"))
            continue
        if line in ("/model", "/switch", "/handoff"):
            sess.persist()
            return "switch"
        if line == "/compact":
            _do_compact(sess)
            continue
        if line == "/save":
            sess.persist()
            print(_dim(f"  saved {sess.session_id} · {title_from(sess.history)}"))
            continue
        if line.startswith("/resume"):
            rec = _pick_resume()
            if rec:
                apply_record(sess, rec)
                src = rec.get("source") or {}
                if src.get("kind") and src.get("kind") != getattr(sess.provider, "kind", ""):
                    print(_dim("  session was on a different backend — /handoff to switch, or continue here"))
            continue
        if line == "/skills":
            if not sess.skills:
                print(_dim("  no skills. /import Claude or Codex, or add .lightlx/skills/<name>/SKILL.md"))
            for s in sorted(sess.skills.values(), key=lambda x: x.name):
                fork = " · fork" if s.fork else ""
                print(f"  /{s.name:<22} {_dim(s.source + fork)}  {s.description[:70]}")
            continue
        if line == "/import":
            _import_menu(sess)
            continue
        if line == "/memory":
            files = sess.memory.list_files()
            print(_dim(f"  {sess.memory.dir}"))
            if files:
                for name in files:
                    print(f"  {name}")
            else:
                print(_dim("  empty — the agent writes here as it learns, or ask it to remember something"))
            idx = sess.memory.load_index()
            if idx:
                print(_dim("\n  MEMORY.md"))
                print(idx[:1500])
            continue
        if line == "/init":
            print(_dim("  " + init_project(sess.ws.root)))
            sess.reload_knowledge()
            continue
        if line.startswith("/"):
            name, _, rest = line[1:].partition(" ")
            sk = sess.skills.get(name) or sess.skills.get(name.lower())
            if sk and sk.user_invocable:
                printer.reset()
                body = expand_skill(sk, rest, workspace=str(sess.ws.root))
                if sk.fork:
                    run_turn(sess, f"Run this skill as a subagent (task tool, type {sk.agent_kind}):\n\n{body}", printer)
                else:
                    run_turn(sess, body, printer)
                continue
            print(_dim(f"  unknown command {line} — try /help or /skills"))
            continue
        printer.reset()
        run_turn(sess, line, printer)
