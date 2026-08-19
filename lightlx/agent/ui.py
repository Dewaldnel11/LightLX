import re
import shutil
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from .context import estimate_tokens

_ANSI = re.compile(r"\033\[[0-9;]*m")
_BG = "\033[48;5;236m\033[38;5;252m"
_RESET = "\033[0m"


def vislen(text: str) -> int:
    return len(_ANSI.sub("", text or ""))


def fmt_tok(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def ctx_usage(sess):
    used = estimate_tokens(getattr(sess, "history", None) or [])
    ctx = (
        getattr(sess, "context_length", None)
        or getattr(sess, "ctx_limit", None)
        or 0
    )
    try:
        ctx = int(ctx or 0)
    except (TypeError, ValueError):
        ctx = 0
    pct = min(100, int(round(100 * used / ctx))) if ctx else 0
    return used, ctx, pct


def meter(pct, width=8) -> str:
    pct = max(0, min(100, int(pct or 0)))
    fill = int(round(pct / 100 * width))
    fill = max(0, min(width, fill))
    return "█" * fill + "░" * (width - fill)


def _pct_color(pct) -> str:
    if pct >= 90:
        return "\033[38;5;203m"
    if pct >= 70:
        return "\033[38;5;178m"
    return "\033[38;5;114m"


def short_model(sess) -> str:
    provider = getattr(sess, "provider", None)
    label = getattr(provider, "label", None) or getattr(sess, "name", "") or ""
    for prefix in ("lmstudio/", "ollama/", "openai/", "mlx/"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    return label or "model"


def _workspace_name(sess) -> str:
    ws = getattr(sess, "ws", None)
    root = getattr(ws, "root", None) if ws is not None else None
    if not root:
        return ""
    return Path(str(root)).name


def format_bar(sess, cols=80, busy=False, color=True) -> str:
    cols = max(20, int(cols or 80))
    used, ctx, pct = ctx_usage(sess)
    m = meter(pct)
    if color:
        ctx_bit = f"{_pct_color(pct)}{m}{_BG} {fmt_tok(used)}"
    else:
        ctx_bit = f"{m} {fmt_tok(used)}"
    if ctx:
        ctx_bit += f"/{fmt_tok(ctx)} {pct}%"
    model = short_model(sess)
    bits = [ctx_bit, model]
    if hasattr(sess, "native_tools"):
        caps = ((getattr(sess, "_runtime", None) or {}).get("caps") or {})
        if caps.get("tools") is False:
            bits.append("no-tools")
        elif not sess.native_tools:
            bits.append("text")
        elif not getattr(sess, "allow_subagents", True):
            bits.append("inline")
        else:
            bits.append("native")
    mode = getattr(sess, "mode", "") or ""
    think = getattr(sess, "think", False)
    if mode:
        bits.append(mode + ("·think" if think else ""))
    last = getattr(sess, "last_turn", None) or {}
    if busy:
        bits.append("working…")
    elif last.get("steps") is not None:
        bits.append(f"last {last['steps']} · {last.get('dur') or '0s'}")
    else:
        cap = int(getattr(sess, "max_tokens", 0) or 0)
        if cap > 0:
            bits.append(f"max {fmt_tok(cap)}")
        else:
            room = max((ctx or 0) - used, 0)
            bits.append(f"max {fmt_tok(room or ctx or 0)}")
    ws = _workspace_name(sess)
    if ws:
        bits.append(ws)
    line = " " + "  ·  ".join(bits) + " "
    raw = _ANSI.sub("", line)
    if len(raw) > cols:
        keep = cols - 1
        out, n = [], 0
        i = 0
        while i < len(line) and n < keep:
            if line[i] == "\033":
                m = _ANSI.match(line, i)
                if m:
                    out.append(m.group(0))
                    i = m.end()
                    continue
            out.append(line[i])
            n += 1
            i += 1
        line = "".join(out) + "…"
        raw = _ANSI.sub("", line)
    pad = max(0, cols - len(raw))
    line = line + (" " * pad)
    if not color:
        return _ANSI.sub("", line)
    return _BG + line + _RESET


def fallback_prompt(sess) -> str:
    cols = shutil.get_terminal_size((80, 24)).columns
    return "\n" + format_bar(sess, cols, color=False).rstrip() + "\n\033[1m›\033[0m "


class ChatBar:
    """Sticky last-line status; input sits on the line above it."""

    def __init__(self):
        self.enabled = bool(sys.stdout.isatty() and sys.stdin.isatty())
        self._on = False
        self.sess = None
        self.busy = False
        self._lock = threading.Lock()
        self._prev_winch = None

    def attach(self, sess):
        self.sess = sess

    def size(self):
        sz = shutil.get_terminal_size((80, 24))
        return max(20, sz.columns), max(6, sz.lines)

    def start(self):
        if not self.enabled:
            return
        self._on = True
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._prev_winch = signal.getsignal(signal.SIGWINCH)
        try:
            signal.signal(signal.SIGWINCH, self._winch)
        except Exception:
            self._prev_winch = None
        self._set_scroll()
        self.paint()

    def stop(self):
        if not self._on:
            return
        self._on = False
        if self._prev_winch is not None:
            try:
                signal.signal(signal.SIGWINCH, self._prev_winch)
            except Exception:
                pass
        cols, rows = self.size()
        sys.stdout.write("\033[r")
        sys.stdout.write(f"\033[{rows};1H\033[2K\033[0m\033[?25h\n")
        sys.stdout.flush()

    def _set_scroll(self):
        _, rows = self.size()
        sys.stdout.write(f"\033[1;{max(1, rows - 1)}r")
        sys.stdout.flush()

    def _winch(self, signum, frame):
        if self._on:
            self._set_scroll()
            self.refresh()
        prev = self._prev_winch
        if callable(prev):
            prev(signum, frame)

    def paint(self):
        if not self._on or self.sess is None:
            return
        with self._lock:
            cols, rows = self.size()
            bar = format_bar(self.sess, cols, busy=self.busy, color=True)
            sys.stdout.write(f"\033[{rows};1H{bar}")
            sys.stdout.flush()

    def refresh(self):
        if not self._on:
            return
        sys.stdout.write("\033[s")
        self.paint()
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def readline(self) -> str:
        if not self.enabled or not self._on:
            return input(fallback_prompt(self.sess) if self.sess else "› ")
        self.busy = False
        cols, rows = self.size()
        self.paint()
        sys.stdout.write(f"\033[{rows - 1};1H\033[2K\033[0m")
        sys.stdout.flush()
        return input("\033[1m›\033[0m ")

    def suspend(self):
        if not self._on:
            return
        sys.stdout.write("\033[r\033[0m")
        _, rows = self.size()
        sys.stdout.write(f"\033[{rows};1H\033[2K\n")
        sys.stdout.flush()

    def resume(self):
        if not self.enabled or not self._on:
            return
        sys.stdout.write("\n")
        self._set_scroll()
        self.paint()

    @contextmanager
    def paused(self):
        self.suspend()
        try:
            yield
        finally:
            self.resume()
