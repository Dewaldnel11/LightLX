import os
import sys
import time

from .context import detect_context, estimate_tokens, handoff_note, maybe_compact
from .loop import SubagentLauncher, _fmt_dur, _git_snapshot, run_agent, seed_messages
from .ui import ChatBar
from .providers import refresh_remote_provider, runtime_notice
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
  /tokens auto|N  reply cap (auto = remaining context)
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
        self.allow_subagents = True
        self._runtime = None
        self.source = source or {}
        self.history = []
        self.pending = None
        self.session_id = None
        self.last_turn = None
        self.hub = MCPHub()
        self.temperature = float(prefs.get("temperature") or 0.1)
        self.context_length = detect_context(provider)
        raw_max = prefs.get("agent_max_tokens", 0)
        try:
            raw_max = int(raw_max)
        except (TypeError, ValueError):
            raw_max = 0
        # 4096 was the old built-in cap; 0 = use the model's remaining context.
        self.max_tokens = 0 if raw_max in (0, 4096) else max(raw_max, 0)
        self.memory = MemoryStore(self.ws.root)
        self.skills = discover_skills(self.ws.root)
        self.instructions = load_instructions(self.ws.root)
        self._connect_mcp()
        self._notices = self.refresh_runtime(announce=True)

    def apply_runtime(self, runtime, announce=False):
        if not runtime:
            return []
        old = self._runtime
        caps = runtime.get("caps") or {}
        model_changed = bool(old and old.get("model") and runtime.get("model") and old["model"] != runtime["model"])
        self.allow_subagents = bool(caps.get("subagents", True))
        if caps.get("tools") is False:
            self.native_tools = False
        elif model_changed:
            self.native_tools = True
        ctx = caps.get("context")
        if ctx:
            self.context_length = int(ctx)
        notes = runtime_notice(old, runtime) if announce else []
        self._runtime = runtime
        return notes

    def refresh_runtime(self, announce=False):
        try:
            runtime = refresh_remote_provider(self.provider)
        except Exception:
            runtime = None
        return self.apply_runtime(runtime, announce=announce)

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
        include_task = depth < 1 and kind != "explore" and self.allow_subagents
        readonly = kind == "explore"
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

    def on_event(kind, text="", name="", detail="", depth=0, ok=True, **kw):
        pad = "  " + ("  " * depth)
        key = kw.get("key") or name
        if kind == "text":
            if not started["n"]:
                print()
                started["n"] = True
            chunk = text
            if depth >= 1:
                chunk = _dim(text)
            sys.stdout.write(chunk)
            sys.stdout.flush()
        elif kind == "tool_start":
            if started["n"]:
                print()
                started["n"] = False
            tool_stack[key] = {"detail": detail or name, "t0": time.time(), "name": name}
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
            info = tool_stack.pop(key, {})
            dur = time.time() - info.get("t0", time.time())
            mark = _ok("✓") if ok else _err("✗")
            tail = f" {_dim(f'{dur:.1f}s')}" if dur > 0.2 else ""
            shown = info.get("name") or name
            if shown in ("read_file", "list_dir", "glob", "grep", "fetch_url", "docs"):
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
            subagent_stack[key] = detail
        elif kind == "subagent_end":
            if started["n"]:
                print()
                started["n"] = False
            detail = subagent_stack.pop(key, detail)
            mark = _ok("✓") if ok else _err("✗")
            short = detail[:60] + ("…" if len(detail) > 60 else "")
            status = kw.get("status") or ""
            steps = kw.get("steps")
            dur = kw.get("dur")
            extra = ""
            if status and status != "done":
                extra = f" — {status.replace('_', ' ')}"
                bits = []
                if steps:
                    bits.append(f"{steps} steps")
                if dur is not None:
                    bits.append(_fmt_dur(dur))
                if bits:
                    extra += f" ({', '.join(bits)})"
            elif steps:
                extra = f" ({steps} steps"
                if dur is not None:
                    extra += f", {_fmt_dur(dur)}"
                extra += ")"
            print(_dim(f"{pad}{mark} subagent: {short}{extra}"))
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
        include_task = depth < 1 and kind != "explore" and getattr(sess, "allow_subagents", True)
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


def _tools_http_rejected(err):
    msg = str(err).lstrip()
    return msg.startswith("400")


def _finish_turn(sess, result, dur=None):
    sess.pending = None
    sess.history = [m for m in (result.messages or []) if m.get("role") != "system"]
    sess.last_turn = {
        "steps": getattr(result, "steps", 0),
        "dur": _fmt_dur(dur or 0),
        "status": getattr(result, "status", "done"),
    }
    if sess.native_tools:
        used = any(m.get("role") == "tool" for m in result.messages or [])
        if used:
            sess._idle_native = 0
        else:
            sess._idle_native = getattr(sess, "_idle_native", 0) + 1
            if sess._idle_native >= 2:
                sess.native_tools = False
                print(_dim("  no tool calls in native mode — switching to text tool-call mode"))
    write_tools = {"write_file", "edit_file", "bash"}
    wrote = False
    for m in result.messages or []:
        for tc in m.get("tool_calls") or []:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
            if name in write_tools:
                wrote = True
                break
        if wrote:
            break
    if wrote:
        snap = _git_snapshot(str(sess.ws.root))
        if snap and snap[0]:
            line = snap[0].replace("\n", ", ")
            if len(line) > 120:
                line = line[:117] + "…"
            print(_dim(f"  changed: {line}"))
    sess.persist()


def resume_pending(sess, on_event):
    for line in sess.refresh_runtime(announce=True):
        print(_dim("  " + line))
    registry = _bind_events(sess, on_event)
    rest = hydrate_messages(sess.pending)
    messages = seed_messages(
        str(sess.ws.root), registry(kind="general", depth=0),
        rest, tools_as_text=not sess.native_tools,
        extra=sess.prompt_extra(), subagents=sess.allow_subagents,
    )
    print(_dim("  resuming in-flight turn…"))
    t0 = time.time()
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
        if sess.native_tools and _tools_http_rejected(e):
            sess.native_tools = False
            print(_dim("  model rejected the tools parameter — switching to text tool-call mode"))
            return resume_pending(sess, on_event)
        print(_dim(f"\n  {e}"))
        return
    _finish_turn(sess, result, dur=time.time() - t0)


def run_turn(sess, user_text, on_event):
    for line in sess.refresh_runtime(announce=True):
        print(_dim("  " + line))
    registry = _bind_events(sess, on_event)
    sess.history.append({"role": "user", "content": user_text})
    sess.history, _ = maybe_compact(
        sess.provider, sess.history, sess.context_length, sess.max_tokens, on_event=on_event,
    )
    messages = seed_messages(
        str(sess.ws.root), registry(kind="general", depth=0),
        sess.history, tools_as_text=not sess.native_tools,
        extra=sess.prompt_extra(), subagents=sess.allow_subagents,
    )
    sess.checkpoint(messages)
    t0 = time.time()
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
        if sess.native_tools and _tools_http_rejected(e):
            if sess.history and sess.history[-1].get("role") == "user":
                sess.history.pop()
            sess.pending = None
            sess.native_tools = False
            print(_dim("  model rejected the tools parameter — switching to text tool-call mode"))
            return run_turn(sess, user_text, on_event)
        print(_dim(f"\n  {e}"))
        sess.history.pop()
        sess.pending = None
        sess.persist()
        return
    _finish_turn(sess, result, dur=time.time() - t0)


def _prompt_line(sess):
    from .ui import fallback_prompt
    return fallback_prompt(sess)


def settings_menu(sess):
    while True:
        print("\n  " + _bold("Settings") + _dim("   number · Enter to back"))
        print(f"   1  Reply length   {_bold('auto' if not sess.max_tokens else str(sess.max_tokens))}")
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
                val = input(_dim(f"  tokens (now {'auto' if not sess.max_tokens else sess.max_tokens}; 0 = auto) › ")).strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if val.lower() in ("auto", "max", "0"):
                sess.max_tokens = sess.prefs["agent_max_tokens"] = 0
                print(_dim("  reply length = auto (model window)"))
            elif val.isdigit() and int(val) > 0:
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
    rows = list_sessions(15, one_per_project=True)
    if not rows:
        print(_dim("  no saved sessions"))
        return None
    print("\n  " + _bold("Sessions"))
    for i, s in enumerate(rows, 1):
        label = project_name(s)
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
    for line in getattr(sess, "_notices", []) or []:
        print(_dim("  " + line))
    sess._notices = []
    print(_dim("  /menu · /skills · /memory · /help · /exit"))
    printer = make_printer()
    bar = ChatBar()
    bar.attach(sess)

    def on_event(kind, **kw):
        printer(kind, **kw)
        if kind in ("tool_end", "subagent_end", "turn_end", "compact_done", "error"):
            bar.refresh()

    on_event.reset = printer.reset
    bar.start()
    try:
        if sess.pending:
            bar.busy = True
            bar.refresh()
            resume_pending(sess, on_event)
            bar.busy = False
            bar.refresh()
        while True:
            try:
                line = bar.readline().strip()
            except (EOFError, KeyboardInterrupt):
                sess.persist()
                return
            if not line:
                continue
            if line in ("/exit", "/quit", "exit", "quit", "/q"):
                sess.persist()
                return
            if line in ("/menu", "/", "/settings"):
                with bar.paused():
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
                sess.last_turn = None
                print(_dim("  conversation cleared — new session"))
                bar.refresh()
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
                if not sess.allow_subagents:
                    print(_dim("  this model cannot run nested subagents — work stays in this chat (inline)"))
                else:
                    print(_dim("  ask the model to use the task tool — several in one turn run in parallel"))
                    print(_dim("  types: explore (read-only) · implement · general"))
                continue
            if line.startswith("/workspace"):
                parts = line.split(maxsplit=1)
                with bar.paused():
                    _set_workspace(sess, parts[1] if len(parts) == 2 else None)
                continue
            if line.startswith("/tokens"):
                parts = line.split()
                if len(parts) == 2 and parts[1].lower() in ("auto", "max", "0"):
                    sess.max_tokens = sess.prefs["agent_max_tokens"] = 0
                    print(_dim("  reply length = auto (model window)"))
                elif len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                    sess.max_tokens = sess.prefs["agent_max_tokens"] = int(parts[1])
                    print(_dim(f"  reply length = {sess.max_tokens}"))
                else:
                    print(_dim("  auto (model window)" if not sess.max_tokens else str(sess.max_tokens)))
                bar.refresh()
                continue
            if line in ("/model", "/switch", "/handoff"):
                sess.persist()
                return "switch"
            if line == "/compact":
                _do_compact(sess)
                bar.refresh()
                continue
            if line == "/save":
                sess.persist()
                print(_dim(f"  saved {sess.session_id} · {title_from(sess.history)}"))
                continue
            if line.startswith("/resume"):
                with bar.paused():
                    rec = _pick_resume()
                    if rec:
                        apply_record(sess, rec)
                        src = rec.get("source") or {}
                        if src.get("kind") and src.get("kind") != getattr(sess.provider, "kind", ""):
                            print(_dim("  session was on a different backend — /handoff to switch, or continue here"))
                bar.attach(sess)
                bar.refresh()
                continue
            if line == "/skills":
                if not sess.skills:
                    print(_dim("  no skills. /import Claude or Codex, or add .lightlx/skills/<name>/SKILL.md"))
                for s in sorted(sess.skills.values(), key=lambda x: x.name):
                    fork = " · fork" if s.fork else ""
                    print(f"  /{s.name:<22} {_dim(s.source + fork)}  {s.description[:70]}")
                continue
            if line == "/import":
                with bar.paused():
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
                    bar.busy = True
                    bar.refresh()
                    if sk.fork and sess.allow_subagents:
                        run_turn(sess, f"Run this skill as a subagent (task tool, type {sk.agent_kind}):\n\n{body}", on_event)
                    else:
                        if sk.fork and not sess.allow_subagents:
                            print(_dim("  subagents unavailable — running this skill inline"))
                        run_turn(sess, body, on_event)
                    bar.busy = False
                    bar.refresh()
                    continue
                print(_dim(f"  unknown command {line} — try /help or /skills"))
                continue
            printer.reset()
            bar.busy = True
            bar.refresh()
            run_turn(sess, line, on_event)
            bar.busy = False
            bar.refresh()
    finally:
        bar.stop()
