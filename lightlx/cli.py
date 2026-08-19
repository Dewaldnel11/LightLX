# LightLX CLI — fully input-driven. No flags:
#
#   lightlx          → menus and questions, then chat / agent

import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .state import add_recent, add_recent_source, load_state, save_state
from .agent.ui import ChatBar

BANNER = r"""
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
"""

HELP = """commands
  /menu          settings — reasoning, reply length, switch model
  /think         toggle reasoning (deeper answers, much slower)
  /tokens N      set max tokens per reply
  /clear         forget the conversation, start fresh
  /model         load a different model / backend (Ollama, LM Studio, …)
  /agent         switch this local model into the full agent (tools + MCP)
  /fast          GLM-5.2 only — 4-bit skeleton, reloads (~1.2× faster)
  /help          show this
  /exit          quit
type anything else to send it to the model   ·   Ctrl-C stops a reply"""


def _dim(s):
    return f"\033[2m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def nice_name(path):
    return os.path.basename(path.rstrip("/")) or path


def _shorten(path, width=46):
    p = path.replace(os.path.expanduser("~"), "~")
    return p if len(p) <= width else "…" + p[-(width - 1):]


# ---------------------------------------------------------------- status line

class StatusLine:
    """Animated single-line status that rides as a suffix after streamed text."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.text = ""
        self.frame = 0
        self.running = False
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            with self.lock:
                self.frame = (self.frame + 1) % len(self.FRAMES)
                self._draw()
            time.sleep(0.1)

    def _draw(self):
        s = f" {self.FRAMES[self.frame]} {self.text}"
        sys.stdout.write("\033[K" + s + f"\033[{len(s)}D")
        sys.stdout.flush()

    def update(self, text):
        with self.lock:
            self.text = text

    def emit(self, text):
        with self.lock:
            if self.enabled:
                sys.stdout.write("\033[K")
            sys.stdout.write(text)
            sys.stdout.flush()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
        with self.lock:
            if self.enabled:
                sys.stdout.write("\033[K")
                sys.stdout.flush()


# ---------------------------------------------------------------- model paths

def clean_path(p: str) -> str:
    p = p.strip()
    if p.startswith("@"):
        p = p[1:].strip()
    p = p.strip("'\"").strip()
    p = p.replace("\\ ", " ")  # shell drag-and-drop escapes spaces
    return os.path.expanduser(p)


def is_model_dir(d: str) -> bool:
    if not d or not Path(d, "config.json").exists():
        return False
    p = Path(d)
    return (p / "model.safetensors.index.json").exists() or any(p.glob("*.safetensors"))


def pick_model(state) -> str | None:
    """Interactive picker: choose a remembered model by number, or drag/paste a
    folder. Returns a valid model dir, or None if the user quits."""
    recent = [p for p in state.get("recent_models", []) if is_model_dir(p)]
    if recent:
        print("\nRecent models")
        for i, p in enumerate(recent, 1):
            print(f"  {i}  {nice_name(p):<26} {_dim(_shorten(p))}")
        print(_dim("\nPick a number — or drag in / paste a model folder.  (q to quit)"))
    else:
        print("\nDrag a model folder here, or paste its path to begin.  " + _dim("(q to quit)"))
        print(_dim("a model folder contains config.json and .safetensors weights"))
    while True:
        try:
            raw = input("\n" + _bold("›") + " ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if not raw:
            continue
        if raw.isdigit() and recent and 1 <= int(raw) <= len(recent):
            return recent[int(raw) - 1]
        d = clean_path(raw)
        if is_model_dir(d):
            return d
        print(_dim(f"  not a model folder: {d}  — pick a number, paste a valid path, or q"))


# ---------------------------------------------------------------- model build

def build_model(model_dir, max_layers, verbose, expert_cache_gb, pin_attn_layers, wired_gb,
                skeleton_bits, prefetch=False, force_stream=False):
    from mlx_lm.models.glm_moe_dsa import ModelArgs
    from .model import StreamingGLM, _total_ram_gb
    cfg = json.load(open(Path(model_dir) / "config.json"))
    mtype = cfg.get("model_type")
    size_gb = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors")) / 1e9
    fits = size_gb < 0.65 * _total_ram_gb()  # leave room for KV cache, activations, OS

    if fits and not force_stream and not max_layers:          # fits in RAM → resident (fast)
        from .generic import ResidentModel
        if verbose:
            print(f"  {size_gb:.1f} GB — fits in memory, loading resident (fast)")
        return ResidentModel(model_dir, verbose=verbose)

    if verbose:                                               # too big → stream from disk
        why = "forced" if force_stream else ("debug" if max_layers else f"{size_gb:.0f} GB > memory")
        print(f"  {why} — streaming from disk (slow but it runs)")
    if mtype == "glm_moe_dsa":
        args = ModelArgs.from_dict(cfg)
        if max_layers:
            args.num_hidden_layers = min(max_layers, args.num_hidden_layers)
        return StreamingGLM(model_dir, args, verbose=verbose, expert_cache_gb=expert_cache_gb,
                            pin_attn_layers=pin_attn_layers, wired_gb=wired_gb, skeleton_bits=skeleton_bits,
                            prefetch=prefetch)
    from .generic import GenericStreamingModel
    return GenericStreamingModel(model_dir, verbose=verbose, max_layers=max_layers)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def model_arch(model_dir):
    return json.load(open(Path(model_dir) / "config.json")).get("model_type", "?")


# ---------------------------------------------------------------- generation

def _fmt_eta(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s/60:.1f}m"


def _encode(tok, messages, think):
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_dict=True, enable_thinking=think)
    except TypeError:  # tokenizer built without the enable_thinking kwarg
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def generate(model, tok, eos, messages, max_tokens, verbose, think=False, ctx_limit=8192, on_token=None):
    # `messages` is the whole conversation. Drop oldest turns if prompt + reply budget
    # would overflow the context window, so multi-turn memory stays within bounds.
    msgs = list(messages)
    ids = _encode(tok, msgs, think)
    while len(ids) + max_tokens > ctx_limit and len(msgs) > 1:
        msgs = msgs[2:] if len(msgs) > 2 else msgs[-1:]  # drop an oldest user/assistant pair
        ids = _encode(tok, msgs, think)
    cache = model.make_cache()

    sl = StatusLine(enabled=verbose)
    state = {"phase": "prefill", "tok": 0, "win": deque(maxlen=10)}

    def on_layer(done, total, gb_read, elapsed):
        win = state["win"]
        win.append((elapsed, gb_read))
        e0, b0 = win[0]
        de, db = elapsed - e0, gb_read - b0
        if de > 1e-3:
            gbps = (db / 1e9) / de
            eta = (de / max(len(win) - 1, 1)) * (total - done)
            state["last"] = (gbps, eta)
        else:
            # window just reset (layer 1 of a token) -> reuse the previous reading
            # instead of flashing 0.00 GB/s · ~0s
            gbps, eta = state.get("last", (0.0, 0.0))
        head = "prefill" if state["phase"] == "prefill" else f"tok {state['tok']}/{max_tokens}"
        sl.update(f"{head} · layer {done}/{total} · {gbps:.2f} GB/s · ~{_fmt_eta(eta)} left")

    streaming = getattr(model, "streaming", True)
    label = "reasoning — long" if think else "direct"
    if verbose:
        print(_dim(f"\n{'streaming from disk · ' if streaming else 'resident · '}{label}\n"))
    sl.start()
    sl.update("thinking…")
    t0 = time.time()
    n = 0
    gen_ids = []
    import mlx.core as mx
    try:
        state["win"] = deque(maxlen=10)
        logits = model(mx.array([ids]), cache, on_layer=on_layer)
        state["phase"] = "decode"
        for _ in range(max_tokens):
            nxt = int(mx.argmax(logits[0, -1]))
            if nxt in eos:
                break
            piece = tok.decode([nxt])
            if on_token:
                on_token(piece)
            else:
                sl.emit(piece)
            gen_ids.append(nxt)
            n += 1
            state["tok"] = n
            state["win"] = deque(maxlen=10)
            logits = model(mx.array([[nxt]]), cache, on_layer=on_layer)
    except KeyboardInterrupt:
        sl.stop()
        print(_dim("\n— stopped —"))
        return tok.decode(gen_ids).strip() if gen_ids else ""  # keep partial reply in history
    sl.stop()
    dt = time.time() - t0
    if verbose:
        extra = f" · {model.w.bytes_read/1e9:.0f} GB read" if streaming and model.w.bytes_read else ""
        rate = n / max(dt, 1e-9)
        speed = f"{rate:.2f} tok/s" if rate >= 0.01 else f"{dt/max(n,1):.0f}s/token"  # s/tok for slow streamed runs
        print(_dim(f"\n\n  {n} tokens · {dt:.0f}s · {speed}{extra}"))
    elif not on_token:
        print()
    return tok.decode(gen_ids).strip()


# ---------------------------------------------------------------- session

class Session:
    def __init__(self, state):
        self.state = state
        prefs = state.get("prefs") or {}
        self._pref = {"think": bool(prefs.get("think", False)),
                      "max_tokens": int(prefs.get("max_tokens", 512)),
                      "fast": bool(prefs.get("fast", False)),
                      "stream": bool(prefs.get("stream", False))}
        self.think = self._pref["think"]
        self.max_tokens = self._pref["max_tokens"]
        self.fast = self._pref["fast"]
        self.force_stream = self._pref["stream"]
        self.model = self.tok = self.eos = None
        self.model_dir = self.name = self.arch = None
        self.history = []
        self._pending_source = None

    def load(self, model_dir, announce=True):
        if announce:
            print(f"\nloading {_bold(nice_name(model_dir))} …")
        self.model = build_model(
            model_dir, None, True,
            float(self.state["prefs"].get("expert_cache_gb") or 0),
            int(self.state["prefs"].get("pin_attn_layers") or 0),
            self.state["prefs"].get("wired_gb"),
            skeleton_bits=(4 if self.fast else None),
            prefetch=bool(self.state["prefs"].get("prefetch")),
            force_stream=self.force_stream,
        )
        self.tok = load_tokenizer(model_dir)
        eos = json.load(open(Path(model_dir) / "config.json")).get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.model_dir = model_dir
        self.name = nice_name(model_dir)
        self.arch = model_arch(model_dir)
        self.history = []  # fresh conversation per loaded model
        add_recent(self.state, model_dir)
        add_recent_source(self.state, "mlx", model_dir, self.name)
        self.persist()

    @property
    def mode(self):
        return "resident" if not getattr(self.model, "streaming", True) else "streaming"

    @property
    def ctx_limit(self):
        # Use the model's full context window (from config.json) rather than a
        # hardcoded 8k ceiling. GLM's DSA streaming attention keeps a floor at
        # its 2048 index_topk bound; everything else uses the advertised window.
        cfg = {}
        if self.model_dir:
            try:
                cfg = json.load(open(Path(self.model_dir) / "config.json"))
            except Exception:
                cfg = {}
        n = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length")
        try:
            n = int(n) if n else 0
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            n = 32768
        return max(2048, n) if self.is_glm else n

    @property
    def is_glm(self):
        return self.arch == "glm_moe_dsa"

    def persist(self):
        self.state["prefs"] = {**self.state.get("prefs", {}), **self._pref}
        save_state(self.state)


# ---------------------------------------------------------------- REPL + menus

def _prompt_line(sess):
    from .agent.ui import fallback_prompt
    return fallback_prompt(sess)


def set_tokens(sess, arg=None):
    val = arg
    if val is None:
        try:
            val = input(_dim(f"  reply length in tokens (now {sess.max_tokens}) › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
    if val.isdigit() and int(val) > 0:
        sess.max_tokens = sess._pref["max_tokens"] = int(val)
        sess.persist()
        print(_dim(f"  reply length = {sess.max_tokens} tokens"))
    else:
        print(_dim("  enter a positive number"))


def toggle_fast(sess):
    if not sess.is_glm:
        print(_dim("  /fast is GLM-5.2 only (it 4-bit-quantizes the skeleton)"))
        return
    sess.fast = not sess.fast
    print(_dim(f"  switching to {'fast (4-bit skeleton)' if sess.fast else 'full (BF16)'} — reloading, ~30–60s…"))
    try:
        sess.load(sess.model_dir, announce=False)
        print(_dim(f"  now in {'fast' if sess.fast else 'full'} mode"))
    except Exception as e:
        sess.fast = not sess.fast
        print(_dim(f"  reload failed ({e})"))


def switch_model(sess) -> bool:
    from .agent.discover import pick_source
    src = pick_source(sess.state, is_model_dir, lambda: pick_model(sess.state))
    if src is None:
        return False
    if src.get("kind") != "mlx":
        sess._pending_source = src
        return True
    sess.fast = False
    sess._pending_source = None
    sess.load(src["path"])
    print(_dim(f"  loaded {sess.name} · {sess.mode}"))
    return True


def settings_menu(sess) -> str:
    """Returns 'quit' if the user chose to quit, else '' (back to chat)."""
    while True:
        print("\n  " + _bold("Settings") + _dim("   number to change · Enter for back"))
        print(f"   1  Reasoning      {_bold('on' if sess.think else 'off')}   {_dim('deeper, much slower')}")
        print(f"   2  Reply length   {_bold(str(sess.max_tokens))} {_dim('tokens')}")
        print(f"   3  Switch model   {_dim(sess.name)}")
        print(f"   4  Force stream   {_bold('on' if sess.force_stream else 'off')}   {_dim('even if it fits in RAM')}")
        if sess.is_glm:
            print(f"   5  Fast mode      {_bold('on' if sess.fast else 'off')}   {_dim('4-bit skeleton, reloads')}")
        print(f"   q  Quit LightLX")
        try:
            c = input("  " + _bold("›") + " ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ""
        if c in ("", "b", "back"):
            return ""
        elif c == "1":
            sess.think = sess._pref["think"] = not sess.think
            sess.persist()
        elif c == "2":
            set_tokens(sess)
        elif c == "3":
            if switch_model(sess):
                return "switch" if getattr(sess, "_pending_source", None) else ""
        elif c == "4":
            sess.force_stream = sess._pref["stream"] = not sess.force_stream
            sess.persist()
            print(_dim("  will apply on next load — /model to reload"))
        elif c == "5" and sess.is_glm:
            toggle_fast(sess)
        elif c in ("q", "quit"):
            return "quit"
        else:
            print(_dim("  pick a number"))


def repl(sess):
    from .agent.loop import _fmt_dur
    print(_dim(f"\n  {sess.name} · {sess.mode} · {sess.max_tokens} tokens max · remembers the chat"))
    print(_dim("  message the model, or /menu for settings · /help · /clear · /exit"))
    bar = ChatBar()
    bar.attach(sess)
    bar.start()
    try:
        while True:
            try:
                line = bar.readline().strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue
            if line in ("/exit", "/quit", "exit", "quit", "/q"):
                return
            if line in ("/menu", "/settings"):
                with bar.paused():
                    action = settings_menu(sess)
                if action == "quit":
                    return
                if action == "switch" and getattr(sess, "_pending_source", None):
                    return "switch"
                continue
            if line == "/help":
                print(HELP)
                continue
            if line in ("/clear", "/reset", "/new"):
                sess.history = []
                sess.last_turn = None
                print(_dim("  conversation cleared — fresh start"))
                bar.refresh()
                continue
            if line == "/think":
                sess.think = sess._pref["think"] = not sess.think
                sess.persist()
                print(_dim(f"  reasoning {'on — deeper, much slower' if sess.think else 'off'}"))
                bar.refresh()
                continue
            if line == "/fast":
                with bar.paused():
                    toggle_fast(sess)
                continue
            if line == "/model" or line == "/switch":
                with bar.paused():
                    if switch_model(sess) and getattr(sess, "_pending_source", None):
                        return "switch"
                continue
            if line == "/agent":
                return "agent"
            if line.startswith("/tokens"):
                parts = line.split()
                with bar.paused():
                    set_tokens(sess, parts[1] if len(parts) == 2 else None)
                bar.refresh()
                continue
            if line.startswith("/"):
                print(_dim(f"  unknown command {line} — try /help"))
                continue
            sess.history.append({"role": "user", "content": line})
            bar.busy = True
            bar.refresh()
            t0 = time.time()
            reply = generate(sess.model, sess.tok, sess.eos, sess.history, sess.max_tokens,
                             verbose=True, think=sess.think, ctx_limit=sess.ctx_limit)
            dt = time.time() - t0
            if reply:
                sess.history.append({"role": "assistant", "content": reply})
                ntok = max(1, len(reply) // 4)
                sess.last_turn = {"steps": ntok, "dur": _fmt_dur(dt)}
            else:
                sess.history.pop()
            bar.busy = False
            bar.refresh()
    finally:
        bar.stop()


# ---------------------------------------------------------------- entrypoint

def _remember(state, source):
    if source["kind"] == "mlx":
        add_recent_source(state, "mlx", source["path"], nice_name(source["path"]))
    else:
        add_recent_source(state, source["kind"], source["model"], source["model"], url=source.get("url"))
        if source.get("url") and source["kind"] in ("ollama", "lmstudio"):
            state["prefs"][f"{source['kind']}_url"] = source["url"]
        if source["kind"] == "lmstudio" and source.get("api_key"):
            state["prefs"]["lmstudio_api_key"] = source["api_key"]
    save_state(state)


def _ask_choice(title, options, allow_back=True):
    print("\n  " + _bold(title) + (_dim("   number · Enter to back") if allow_back else ""))
    for i, (label, hint) in enumerate(options, 1):
        extra = f"   {_dim(hint)}" if hint else ""
        print(f"   {i}  {label}{extra}")
    while True:
        try:
            raw = input("  " + _bold("›") + " ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if allow_back and raw in ("", "b", "back", "q"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(_dim("  pick a number"))


def _workspace_for(state, carry):
    if carry is not None:
        if isinstance(carry, dict) and carry.get("workspace") and os.path.isdir(carry["workspace"]):
            return carry["workspace"]
        ws = getattr(getattr(carry, "ws", None), "root", None)
        if ws and os.path.isdir(str(ws)):
            return str(ws)
        saved = state["prefs"].get("workspace")
        if saved and os.path.isdir(saved):
            return saved
        return os.getcwd()
    return _ask_workspace(state, carry)


def _ask_workspace(state, carry=None):
    cwd = os.getcwd()
    saved = state["prefs"].get("workspace") or ""
    carried = ""
    if isinstance(carry, dict):
        carried = carry.get("workspace") or ""
    elif carry is not None:
        ws = getattr(getattr(carry, "ws", None), "root", None)
        carried = str(ws) if ws else ""
    choices = []
    paths = []
    for p, label in ((cwd, "this folder"), (saved, "last used"), (carried, "from session")):
        p = os.path.abspath(os.path.expanduser(p)) if p else ""
        if not p or not os.path.isdir(p) or p in paths:
            continue
        choices.append((p, label))
        paths.append(p)
    choices.append(("other", "type a path"))
    pick = _ask_choice("Workspace", [(c[0] if c[0] != "other" else "somewhere else", c[1]) for c in choices], allow_back=False)
    if pick is None:
        return cwd
    chosen = choices[pick][0]
    if chosen != "other":
        state["prefs"]["workspace"] = chosen
        return chosen
    try:
        raw = input(_dim("  folder › ")).strip()
    except (EOFError, KeyboardInterrupt):
        return cwd
    path = os.path.abspath(os.path.expanduser(clean_path(raw) if raw else cwd))
    if not os.path.isdir(path):
        print(_dim(f"  not a directory — using {cwd}"))
        return cwd
    state["prefs"]["workspace"] = path
    return path


def _ask_mlx_mode(carry):
    if carry:
        return "agent"
    pick = _ask_choice("Use this model as", [
        ("Chat", "just talk"),
        ("Agent", "files, tools, MCP, subagents"),
    ], allow_back=False)
    return "agent" if pick == 1 else "chat"


def _run_agent(provider, state, native_tools=True, carry=None, source=None, workspace=None):
    from .agent.repl import AgentSession, agent_repl, apply_record
    ws = workspace or state["prefs"].get("workspace") or os.getcwd()
    sess = AgentSession(provider, ws, state["prefs"], native_tools=native_tools, source=source)
    old_label = None
    if isinstance(carry, dict) and carry.get("history") is not None:
        apply_record(sess, carry)
        old_label = carry.get("provider")
    elif carry is not None and getattr(carry, "history", None) is not None:
        sess.history = list(carry.history)
        sess.session_id = getattr(carry, "session_id", None)
        old = getattr(carry, "provider", None)
        old_label = getattr(old, "label", None) or getattr(carry, "label", None)
    if old_label and old_label != provider.label and sess.history:
        did = sess.apply_handoff(old_label)
        print(_dim(f"  handed off {old_label} → {provider.label} · ctx {sess.context_length}"
                   + (" · compacted" if did else "")))
    try:
        return agent_repl(sess), sess
    finally:
        sess.persist()
        sess.close()
        save_state(state)


def _mlx_provider(sess):
    from .agent.providers import MlxLocal
    return MlxLocal(
        generate, sess.model, sess.tok, sess.eos,
        think=sess.think, ctx_limit=sess.ctx_limit, name=sess.name,
    )


def main():
    if len(sys.argv) > 1:
        print(_dim("lightlx is menu-driven — extra arguments are ignored. just run:  lightlx"))

    print(BANNER)
    state = load_state()
    carry = None
    from .agent.discover import pick_source
    source = pick_source(state, is_model_dir, lambda: pick_model(state))
    if source is None:
        print("bye.")
        return

    while source:
        if source.get("kind") == "resume":
            rec = source.get("record") or {}
            carry = rec
            source = rec.get("source") or None
            if source is None:
                print(_dim("  session has no backend — pick one"))
                source = pick_source(state, is_model_dir, lambda: pick_model(state))
                continue
        _remember(state, source)
        if source["kind"] != "mlx":
            from .agent.discover import build_provider
            try:
                provider = build_provider(source)
            except Exception as e:
                print(_dim(f"  {e}"))
                return
            ws = _workspace_for(state, carry)
            action, agent_sess = _run_agent(
                provider, state, native_tools=True, carry=carry, source=source, workspace=ws,
            )
            if action == "switch":
                carry = agent_sess
                source = pick_source(state, is_model_dir, lambda: pick_model(state))
                continue
            break

        sess = Session(state)
        sess.load(source["path"])
        mode = _ask_mlx_mode(carry)
        if mode == "agent":
            ws = _workspace_for(state, carry)
            action, agent_sess = _run_agent(
                _mlx_provider(sess), state, native_tools=False, carry=carry, source=source, workspace=ws,
            )
        else:
            action, agent_sess = repl(sess), None
        sess.persist()
        if action == "agent":
            ws = _workspace_for(state, carry)
            action, agent_sess = _run_agent(
                _mlx_provider(sess), state, native_tools=False,
                carry=agent_sess or carry, source=source, workspace=ws,
            )
        if action == "switch":
            carry = agent_sess
            source = getattr(sess, "_pending_source", None)
            if source is None:
                source = pick_source(state, is_model_dir, lambda: pick_model(state))
            continue
        break

    save_state(state)
    print(_dim("\nsaved. see you next time."))


if __name__ == "__main__":
    main()
