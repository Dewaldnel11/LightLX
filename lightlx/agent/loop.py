import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .context import maybe_compact
from .parse import looks_like_tool_narration, parse_text_tool_calls
from .prompts import format_tool_list, system_prompt
from .providers import to_openai_messages
from .tools import summarize_call
from .types import ToolCall

MAX_ITERS = 40
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
    try:
        return spec.handler(**(args or {}))
    except TypeError:
        return spec.handler(**{k: v for k, v in (args or {}).items() if k in (spec.parameters.get("properties") or {})})
    except Exception as e:
        return f"error: {e}\n{traceback.format_exc()[-800:]}"


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
    return results


def run_agent(
    provider,
    messages,
    registry,
    *,
    max_tokens=4096,
    max_iters=MAX_ITERS,
    temperature=0.2,
    on_event=None,
    depth=0,
    native_tools=True,
    context_length=None,
):
    tools = [s.openai_tool() for s in registry.values()] if native_tools else None
    last = ""
    steps = 0
    nudged = False
    narrate_nudge = False
    ctx = context_length or getattr(provider, "context_length", None)
    for _ in range(max_iters):
        steps += 1
        if ctx and depth == 0:
            messages, did = maybe_compact(provider, messages, ctx, max_tokens, on_event=on_event)
            if did and on_event:
                on_event("compact_done", depth=depth)
        if on_event:
            on_event("model_start", depth=depth)

        def on_text(t, _d=depth):
            if on_event:
                on_event("text", text=t, depth=_d)

        try:
            payload = to_openai_messages(messages) if native_tools else messages
            comp = provider.complete(
                payload, tools=tools, max_tokens=max_tokens,
                temperature=temperature, on_text=on_text,
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

        messages.append(_assistant_message(comp))
        last = (comp.content or "").strip()
        if not comp.tool_calls:
            used_tools = any(m.get("role") == "tool" for m in messages)
            if looks_like_tool_narration(last) and not narrate_nudge:
                narrate_nudge = True
                if on_event:
                    on_event("text", text="\n", depth=depth)
                messages.append({
                    "role": "user",
                    "content": "Do not narrate. Call the tools now (read_file / list_dir / grep). No more 'let me read' text.",
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

        if on_event and last:
            on_event("text", text="\n", depth=depth)

        outs = execute_tools(
            comp.tool_calls, registry, on_event=on_event, depth=depth,
            parallel=getattr(provider, "parallel_safe", True),
        )
        for call, out in zip(comp.tool_calls, outs):
            messages.append(_tool_message(call, out))

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
            "explore": "You are a read-only research subagent. Do not edit files or run destructive commands. Return a concise report.",
            "implement": "You are an implementation subagent. Make the changes, then report what you changed and how you verified it.",
            "general": "You are a subagent. Complete the assignment and return a concise report of results, files touched, and anything the parent must know.",
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
                max_tokens=self.max_tokens, max_iters=30,
                temperature=self.temperature, on_event=self.on_event,
                depth=1, native_tools=self.native_tools,
            )
        except Exception as e:
            if self.on_event:
                self.on_event("subagent_end", name=kind, detail=description, ok=False, depth=1)
            return f"subagent failed: {e}"
        if self.on_event:
            self.on_event("subagent_end", name=kind, detail=description, ok=True, depth=1)
        report = result.text or "(no final text)"
        return f"subagent [{kind}] {description} — {result.steps} steps\n{report}"
