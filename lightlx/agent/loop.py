import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .context import completion_tokens, maybe_compact
from .parse import looks_like_tool_narration, parse_text_tool_calls
from .prompts import format_tool_list, system_prompt
from .providers import to_openai_messages
from .tools import summarize_call
from .types import ToolCall

MAX_ITERS = 80
MAX_DEPTH = 1


class AgentResult:
    def __init__(self, text, messages, steps, status="done"):
        self.text = text
        self.messages = messages
        self.steps = steps
        self.status = status


def _assistant_message(comp):
    msg = {"role": "assistant", "content": comp.content or ""}
    if comp.tool_calls:
        msg["tool_calls"] = comp.tool_calls
    return msg


def _tool_message(call: ToolCall, result: str):
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": result if result else "(empty)",
    }


TOOL_ALIASES = {
    "shell": "bash",
    "run_terminal_cmd": "bash",
    "run_command": "bash",
    "execute": "bash",
    "execute_command": "bash",
    "terminal": "bash",
    "cmd": "bash",
    "read": "read_file",
    "cat": "read_file",
    "view": "read_file",
    "view_file": "read_file",
    "get_file": "read_file",
    "open_file": "read_file",
    "write": "write_file",
    "create_file": "write_file",
    "save_file": "write_file",
    "str_replace": "edit_file",
    "strreplace": "edit_file",
    "search_replace": "edit_file",
    "replace_in_file": "edit_file",
    "ls": "list_dir",
    "list": "list_dir",
    "list_files": "list_dir",
    "listdir": "list_dir",
    "find_files": "glob",
    "glob_file_search": "glob",
    "search": "grep",
    "codebase_search": "grep",
    "ripgrep": "grep",
    "spawn_agent": "task",
    "subagent": "task",
    "agent": "task",
    "web_fetch": "fetch_url",
    "webfetch": "fetch_url",
    "fetch": "fetch_url",
}

ARG_ALIASES = {
    "path": ("path", "file_path", "file", "filename", "target_file", "target"),
    "command": ("command", "cmd", "code", "script", "command_line"),
    "old_string": ("old_string", "old_str", "old", "search", "original"),
    "new_string": ("new_string", "new_str", "new", "replace", "replacement"),
    "content": ("content", "text", "body", "contents", "data"),
    "pattern": ("pattern", "query", "regex"),
    "description": ("description", "title", "label"),
    "prompt": ("prompt", "instructions", "message"),
    "subagent_type": ("subagent_type", "type", "agent_type", "kind"),
    "url": ("url", "href", "link"),
    "offset": ("offset", "start_line", "start"),
    "limit": ("limit", "end_line", "count", "max_lines"),
    "workdir": ("workdir", "cwd", "working_directory"),
}


def canonical_tool_name(name):
    n = (name or "").strip()
    if "." in n:
        n = n.rsplit(".", 1)[-1]
    key = n.lower().replace("-", "_")
    return TOOL_ALIASES.get(key, n)


def _lookup(registry, name):
    n = canonical_tool_name(name)
    if n in registry:
        return registry[n]
    lower = {k.lower(): v for k, v in registry.items()}
    return lower.get((n or "").lower()) or lower.get((name or "").lower())


def remap_args(spec, args):
    args = dict(args or {})
    props = (spec.parameters or {}).get("properties") or {}
    for canon, alts in ARG_ALIASES.items():
        if canon not in props:
            continue
        if args.get(canon) not in (None, ""):
            continue
        for a in alts:
            if a != canon and args.get(a) not in (None, ""):
                args[canon] = args[a]
                break
    return args


def normalize_calls(calls, registry):
    for tc in calls or []:
        if not isinstance(tc, ToolCall):
            continue
        spec = _lookup(registry, tc.name)
        if spec:
            tc.name = spec.name
            tc.arguments = remap_args(spec, tc.arguments or {})
        else:
            tc.name = canonical_tool_name(tc.name)
    return calls


def _run_one(spec, args):
    args = args or {}
    try:
        return spec.handler(**(args or {}))
    except TypeError:
        props = (spec.parameters or {}).get("properties") or {}
        filtered = {k: v for k, v in (args or {}).items() if k in props}
        try:
            return spec.handler(**filtered)
        except TypeError as e:
            req = (spec.parameters or {}).get("required") or []
            missing = [r for r in req if not (args or {}).get(r)]
            if missing:
                got = ", ".join(sorted(args or {})) or "(none)"
                return (
                    f"error: {spec.name} missing required argument(s): {', '.join(missing)}. "
                    f"got: {got}. provide all of: {', '.join(req)}."
                )
            return f"error: {e}"
    except Exception as e:
        return f"error: {e}\n{traceback.format_exc()[-800:]}"


def unanswered_tool_calls(messages):
    last_i = -1
    last = None
    for i, m in enumerate(messages or []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_i, last = i, m
    if last is None:
        return []
    have = {m.get("tool_call_id") for m in messages[last_i + 1:] if m.get("role") == "tool"}
    out = []
    for tc in last["tool_calls"]:
        if not isinstance(tc, ToolCall):
            continue
        if tc.id not in have:
            out.append(tc)
    return out


def execute_tools(calls, registry, on_event=None, depth=0, parallel=True):
    results = [None] * len(calls)

    def work(i, call):
        spec = _lookup(registry, call.name)
        if spec:
            call.name = spec.name
            call.arguments = remap_args(spec, call.arguments or {})
        label = summarize_call(call.name, call.arguments)
        if on_event:
            on_event("tool_start", name=call.name, detail=label, depth=depth, key=call.id)
        if spec is None:
            out = f"error: unknown tool {call.name!r}. available: {', '.join(sorted(registry))}"
        else:
            out = _run_one(spec, call.arguments)
        if on_event:
            ok = not str(out).startswith("error:")
            on_event("tool_end", name=call.name, detail=label, ok=ok, depth=depth, key=call.id)
        return i, out

    try:
        if parallel and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                futs = [pool.submit(work, i, c) for i, c in enumerate(calls)]
                for fut in as_completed(futs):
                    i, out = fut.result()
                    results[i] = out
        else:
            for i, c in enumerate(calls):
                _, out = work(i, c)
                results[i] = out
    except KeyboardInterrupt:
        for i, r in enumerate(results):
            if r is None:
                results[i] = "error: interrupted"
    return results


def run_agent(
    provider,
    messages,
    registry,
    *,
    max_tokens=4096,
    max_iters=MAX_ITERS,
    temperature=0.1,
    on_event=None,
    depth=0,
    native_tools=True,
    context_length=None,
    deadline=None,
    on_checkpoint=None,
):
    tools = [s.openai_tool() for s in registry.values()] if native_tools else None
    last = ""
    steps = 0
    nudged = False
    narrate_nudges = 0
    length_nudges = 0
    ctx = context_length or getattr(provider, "context_length", None)
    read_tools = {"read_file", "list_dir", "glob", "grep", "fetch_url", "docs"}
    write_tools = {"write_file", "edit_file", "bash"}
    reads_since_write = 0
    READ_BUDGET = 6

    def checkpoint():
        if on_checkpoint and depth == 0:
            on_checkpoint(messages)

    leftover = unanswered_tool_calls(messages)
    if leftover:
        outs = execute_tools(
            leftover, registry, on_event=on_event, depth=depth,
            parallel=getattr(provider, "parallel_safe", True),
        )
        for call, out in zip(leftover, outs):
            messages.append(_tool_message(call, out))
        checkpoint()

    for _ in range(max_iters):
        steps += 1
        if deadline and time.time() > deadline:
            if on_event:
                on_event("error", text="timed out", depth=depth)
            return AgentResult(last or "(timed out)", messages, steps, status="timeout")
        if ctx and depth == 0:
            messages, did = maybe_compact(provider, messages, ctx, max_tokens, on_event=on_event)
            if did:
                checkpoint()
            if did and on_event:
                on_event("compact_done", depth=depth)
        if on_event:
            on_event("model_start", depth=depth)

        try:
            payload = to_openai_messages(messages) if native_tools else messages
            n_out = completion_tokens(payload, ctx, max_tokens)
            comp = provider.complete(
                payload, tools=tools, max_tokens=n_out,
                temperature=temperature, on_text=None,
            )
        except Exception as e:
            if on_event:
                on_event("error", text=str(e), depth=depth)
            raise

        if not comp.tool_calls and comp.content:
            extra_content, extra_calls = parse_text_tool_calls(comp.content)
            if extra_calls:
                comp.content = extra_content
                comp.tool_calls = extra_calls

        if comp.tool_calls:
            normalize_calls(comp.tool_calls, registry)
            bad = []
            for tc in comp.tool_calls:
                name = getattr(tc, "name", None) or tc.get("name")
                spec = _lookup(registry, name)
                args = tc.arguments if isinstance(tc, ToolCall) else {}
                req = (getattr(spec, "parameters", None) or {}).get("required") or [] if spec else []
                missing = [r for r in req if not (args or {}).get(r)]
                if missing:
                    bad.append(f"{name} (missing {', '.join(missing)})")
            if bad:
                if on_event:
                    on_event("text", text="\n", depth=depth)
                note = (
                    " Your response was also cut off by the token limit — keep arguments shorter."
                    if getattr(comp, "finish", "") == "length" else ""
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your tool call(s) had empty or missing required arguments: " + "; ".join(bad) + ". "
                        "Re-issue the call with all required arguments filled in — e.g. edit_file needs "
                        "path, old_string, new_string; task needs description and prompt." + note
                    ),
                })
                continue

        messages.append(_assistant_message(comp))
        last = (comp.content or "").strip()
        if comp.tool_calls:
            last = ""
            checkpoint()
        elif last and on_event:
            on_event("text", text=last, depth=depth)
        if not comp.tool_calls:
            used_tools = any(m.get("role") == "tool" for m in messages)
            if getattr(comp, "finish", "") == "length" and length_nudges < 2:
                length_nudges += 1
                messages.append({
                    "role": "user",
                    "content": "Your reply was cut off by the token limit. Continue from where it stopped.",
                })
                continue
            if looks_like_tool_narration(last) and narrate_nudges < 3:
                narrate_nudges += 1
                if on_event:
                    on_event("text", text="\n", depth=depth)
                messages.append({
                    "role": "user",
                    "content": (
                        f"STOP NARRATING (strike {narrate_nudges}). "
                        "Never write sentences like 'Now let me add…' or 'I'll read…'. "
                        "Call the tool (read_file / list_dir / grep / edit_file / write_file) NOW. "
                        "No prose before tool calls."
                    ),
                })
                continue
            if not last and used_tools and not nudged:
                nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You already gathered information with tools. "
                        "Now write a clear answer and plan for the user. Do not call tools."
                    ),
                })
                continue
            if on_event:
                on_event("turn_end", depth=depth)
            return AgentResult(last, messages, steps)

        outs = execute_tools(
            comp.tool_calls, registry, on_event=on_event, depth=depth,
            parallel=getattr(provider, "parallel_safe", True),
        )
        for call, out in zip(comp.tool_calls, outs):
            messages.append(_tool_message(call, out))
            if call.name in write_tools:
                reads_since_write = 0
            elif call.name in read_tools:
                reads_since_write += 1
        if reads_since_write >= READ_BUDGET:
            if on_event:
                on_event("text", text="\n", depth=depth)
            messages.append({
                "role": "user",
                "content": (
                    f"READ BUDGET EXCEEDED ({reads_since_write} reads, 0 writes). "
                    "STOP READING. You must now call edit_file or write_file to make changes. "
                    "No more read tools until you produce edits."
                ),
            })
            reads_since_write = 0
            continue
        checkpoint()

    if on_event:
        on_event("error", text=f"stopped after {max_iters} steps", depth=depth)
    return AgentResult(last or f"(stopped after {max_iters} steps)", messages, steps, status="max_iters")


def seed_messages(workspace, registry, history, tools_as_text=False, extra="", subagents=True):
    sys = system_prompt(
        workspace, tools_as_text=tools_as_text,
        tool_list=format_tool_list(registry.values()), extra=extra,
        subagents=subagents,
    )
    msgs = [{"role": "system", "content": sys}]
    msgs.extend(history)
    return msgs


def _fmt_dur(sec):
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    return f"{sec // 60}m{sec % 60:02d}s" if sec % 60 else f"{sec // 60}m"


def _run_epilogue(messages):
    counts = {}
    files = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = getattr(tc, "name", None) or ""
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
            args = getattr(tc, "arguments", None)
            if name in ("write_file", "edit_file") and isinstance(args, dict):
                p = args.get("path")
                if p and str(p) not in files:
                    files.append(str(p))
    parts = [f"{n}×{c}" for n, c in sorted(counts.items())] or ["no tools"]
    if files:
        parts.append("files: " + ", ".join(files))
    return "; ".join(parts)


def _git_snapshot(workspace, timeout=10):
    def git(*args):
        try:
            p = subprocess.run(
                ["git", *args], cwd=workspace or None,
                capture_output=True, text=True, timeout=timeout,
            )
        except Exception:
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    if git("rev-parse", "--is-inside-work-tree") != "true":
        return None
    status = git("status", "--short") or ""
    diff = git("diff", "--stat") or ""
    summary = diff.splitlines()[-1].strip() if diff else ""
    return status, summary


class SubagentLauncher:
    def __init__(self, provider, make_registry, workspace, max_tokens, temperature, on_event, native_tools=True, extra=""):
        self.provider = provider
        self.make_registry = make_registry
        self.workspace = workspace
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.on_event = on_event
        self.native_tools = native_tools
        self.extra = extra

    def __call__(self, description, prompt, subagent_type="general"):
        kind = (subagent_type or "general").lower().strip()
        if kind not in ("explore", "implement", "general"):
            kind = "general"
        registry = self.make_registry(kind=kind, depth=1)
        extra = {
            "explore": (
                "You are a READ-ONLY research subagent. Your report is worthless without tools: "
                "you MUST call read_file / list_dir / glob / grep / fetch_url before answering. "
                "Do not edit files or run destructive commands. Gather concrete facts (file paths, "
                "line numbers, code snippets, function names) and return a dense report the parent "
                "can act on. Never say 'looks good' without having read the actual files."
            ),
            "implement": (
                "You are an IMPLEMENTATION subagent. You MUST make real code changes: call "
                "edit_file / write_file / bash until the task is actually done. A text-only reply "
                "is a failure — you must produce edits. After each edit, VERIFY by re-reading the "
                "changed region and running the project's test/lint command if one exists (bash). "
                "Report exactly which files you changed, what changed, and the verification output."
            ),
            "general": (
                "You are a subagent. You MUST actually do the work with tools: read_file, "
                "edit_file, write_file, bash, grep. A report that describes changes without "
                "showing tool calls is a lie. Call the tools to read, modify, and verify. "
                "Return: files touched, what changed, and how you verified it."
            ),
        }[kind]
        history = [{"role": "user", "content": extra + "\n\n" + prompt}]
        messages = seed_messages(
            self.workspace, registry, history,
            tools_as_text=not self.native_tools, extra=self.extra, subagents=False,
        )
        key = uuid.uuid4().hex[:8]
        t0 = time.time()
        if self.on_event:
            self.on_event("subagent_start", name=kind, detail=description, depth=1, key=key)
        try:
            result = run_agent(
                self.provider, messages, registry,
                max_tokens=self.max_tokens, max_iters=MAX_ITERS,
                temperature=self.temperature, on_event=self.on_event,
                depth=1, native_tools=self.native_tools,
                deadline=time.time() + 600,
            )
        except Exception as e:
            dur = int(time.time() - t0)
            if self.on_event:
                self.on_event(
                    "subagent_end", name=kind, detail=description, ok=False, depth=1,
                    key=key, dur=dur, status="error", steps=0,
                )
            return f"subagent failed: {e}"
        dur = int(time.time() - t0)
        status = result.status or "done"
        if self.on_event:
            self.on_event(
                "subagent_end", name=kind, detail=description, ok=status == "done", depth=1,
                key=key, dur=dur, status=status, steps=result.steps,
            )
        status_txt = {
            "timeout": f"timed out after {_fmt_dur(dur)}",
            "max_iters": f"stopped after {result.steps} steps",
            "error": "failed",
        }.get(status, "done")
        header = (
            f"subagent [{kind}] {description} — {status_txt} · {result.steps} steps · "
            f"{_fmt_dur(dur)} · {_run_epilogue(result.messages)}"
        )
        report = result.text or "(no final text)"
        used_tools = any(m.get("role") == "tool" for m in result.messages)
        if not used_tools:
            report += (
                "\n\nWARNING: this subagent produced NO tool calls. Its claims are UNVERIFIED — "
                "re-read any file it mentions yourself before acting on its suggestions."
            )
        if kind == "implement":
            snap = _git_snapshot(self.workspace)
            if snap:
                st, diffsum = snap
                if st:
                    wt = st.replace("\n", "; ")
                    if diffsum:
                        wt += f" ({diffsum})"
                else:
                    wt = "no changes detected — treat claims as UNVERIFIED"
                report += "\n\nWORKING TREE: " + wt
        return header + "\n" + report
