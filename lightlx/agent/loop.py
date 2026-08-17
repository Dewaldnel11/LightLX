import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .context import maybe_compact
from .parse import looks_like_tool_narration, parse_text_tool_calls
from .prompts import format_tool_list, system_prompt
from .providers import to_openai_messages
from .tools import summarize_call
from .types import ToolCall

MAX_ITERS = 80
MAX_DEPTH = 1


class AgentResult:
    def __init__(self, text, messages, steps):
        self.text = text
        self.messages = messages
        self.steps = steps


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


def _lookup(registry, name):
    if name in registry:
        return registry[name]
    lower = {k.lower(): v for k, v in registry.items()}
    return lower.get((name or "").lower())


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
        label = summarize_call(call.name, call.arguments)
        if on_event:
            on_event("tool_start", name=call.name, detail=label, depth=depth)
        if spec is None:
            out = f"error: unknown tool {call.name!r}. available: {', '.join(sorted(registry))}"
        else:
            out = _run_one(spec, call.arguments)
        if on_event:
            ok = not str(out).startswith("error:")
            on_event("tool_end", name=call.name, detail=label, ok=ok, depth=depth)
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
            return AgentResult(last or "(timed out)", messages, steps)
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
            comp = provider.complete(
                payload, tools=tools, max_tokens=max_tokens,
                temperature=temperature, on_text=None,
            )
        except Exception as e:
            if on_event:
                on_event("error", text=str(e), depth=depth)
            raise

        if native_tools and not comp.tool_calls and comp.content:
            extra_content, extra_calls = parse_text_tool_calls(comp.content)
            if extra_calls:
                comp.content = extra_content
                comp.tool_calls = extra_calls

        if native_tools and comp.tool_calls:
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
                messages.append({
                    "role": "user",
                    "content": (
                        "Your tool call(s) had empty or missing required arguments: " + "; ".join(bad) + ". "
                        "Re-issue the call with all required arguments filled in — e.g. edit_file needs "
                        "path, old_string, new_string; task needs description and prompt."
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
    return AgentResult(last or f"(stopped after {max_iters} steps)", messages, steps)


def seed_messages(workspace, registry, history, tools_as_text=False, extra=""):
    sys = system_prompt(
        workspace, tools_as_text=tools_as_text,
        tool_list=format_tool_list(registry.values()), extra=extra,
    )
    msgs = [{"role": "system", "content": sys}]
    msgs.extend(history)
    return msgs


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
            tools_as_text=not self.native_tools, extra=self.extra,
        )
        if self.on_event:
            self.on_event("subagent_start", name=kind, detail=description, depth=1)
        try:
            result = run_agent(
                self.provider, messages, registry,
                max_tokens=self.max_tokens, max_iters=20,
                temperature=self.temperature, on_event=self.on_event,
                depth=1, native_tools=self.native_tools,
                deadline=time.time() + 600,
            )
        except Exception as e:
            if self.on_event:
                self.on_event("subagent_end", name=kind, detail=description, ok=False, depth=1)
            return f"subagent failed: {e}"
        if self.on_event:
            self.on_event("subagent_end", name=kind, detail=description, ok=True, depth=1)
        report = result.text or "(no final text)"
        used_tools = any(m.get("role") == "tool" for m in result.messages)
        if not used_tools:
            report += (
                "\n\nWARNING: this subagent produced NO tool calls. Its claims are UNVERIFIED — "
                "re-read any file it mentions yourself before acting on its suggestions."
            )
        return f"subagent [{kind}] {description} — {result.steps} steps\n{report}"
