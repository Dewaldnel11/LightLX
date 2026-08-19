import os
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
_IN_BG = "\033[48;5;240m\033[38;5;255m"
_MENU_BG = "\033[48;5;238m\033[38;5;250m"
_MENU_SEL = "\033[48;5;245m\033[38;5;232m"
_RESET = "\033[0m"

FOOTER = 2  # input row + status row


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


SLASH_COMMANDS = [
    ("/help", "this list"),
    ("/kickoff", "map the repo, then a sourced plan"),
    ("/approve", "implement the latest kickoff plan"),
    ("/deny", "discard the latest kickoff plan"),
    ("/brain", "cross-project memory"),
    ("/skills", "list skills"),
    ("/memory", "project auto-memory"),
    ("/menu", "settings"),
    ("/tools", "builtin + MCP tools"),
    ("/task", "subagents (serial on LM Studio)"),
    ("/compact", "summarize older turns"),
    ("/tokens", "reply cap (auto|N)"),
    ("/workspace", "show or set folder"),
    ("/docs", "GitHub docs aliases"),
    ("/mcp", "MCP servers"),
    ("/import", "import Claude/Codex skills"),
    ("/init", "write LIGHTLX.md"),
    ("/resume", "resume a saved session"),
    ("/save", "snapshot now"),
    ("/handoff", "switch model, keep chat"),
    ("/model", "same as /handoff"),
    ("/clear", "forget the conversation"),
    ("/exit", "quit"),
]

_SLASH_NEEDS_SPACE = {
    "/kickoff", "/tokens", "/workspace", "/brain", "/resume", "/import", "/docs",
}

MENU_MAX = 9


def slash_catalog(sess=None):
    items = list(SLASH_COMMANDS)
    seen = {c.lower() for c, _ in items}
    skills = getattr(sess, "skills", None) or {}
    for sk in sorted(skills.values(), key=lambda s: s.name):
        if not getattr(sk, "user_invocable", True):
            continue
        key = "/" + sk.name
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        hint = (getattr(sk, "description", None) or "")[:42]
        items.append((key, hint))
    return items


def filter_slash(items, typed):
    q = typed or ""
    if not q.startswith("/") or " " in q:
        return []
    key = q.lower()
    if key == "/":
        return list(items)[:MENU_MAX]
    hits = [it for it in items if it[0].lower().startswith(key)]
    return hits[:MENU_MAX]


def slash_enter(buf, catalog, sel=0):
    """Enter on the slash palette: (next_buffer, submit).

    Exact `/brain` / `/kickoff` submit. A prefix of a command that takes an
    argument (e.g. `/ki`) expands to `/kickoff ` and stays in the input.
    """
    buf = buf or ""
    hits = filter_slash(catalog, buf)
    if hits and " " not in buf:
        cmd = hits[min(max(sel, 0), len(hits) - 1)][0]
        if buf.lower() == cmd.lower():
            return buf, True
        if buf in ("", "/") or cmd.lower().startswith(buf.lower()):
            if cmd in _SLASH_NEEDS_SPACE:
                return cmd + " ", False
            return cmd, True
    return buf, True


def fallback_prompt(sess) -> str:
    cols = shutil.get_terminal_size((80, 24)).columns
    return "\n" + format_bar(sess, cols, color=False).rstrip() + "\n\033[1m›\033[0m "


def footer_rows(rows, menu_h=0):
    """Return (scroll_bottom, menu_top, input_row, status_row)."""
    rows = max(6, int(rows or 24))
    menu_h = max(0, int(menu_h or 0))
    status_row = rows
    input_row = rows - 1
    menu_top = input_row - menu_h
    scroll_bottom = max(1, menu_top - 1)
    return scroll_bottom, menu_top, input_row, status_row


def format_input_line(buf, cols=80, busy=False) -> str:
    cols = max(20, int(cols or 80))
    if busy:
        text = " ›  "
    else:
        text = " › " + (buf or "")
    vis = _ANSI.sub("", text)
    if len(vis) > cols:
        text = "… " + vis[-(cols - 2):]
        vis = text
    pad = max(0, cols - len(vis))
    return _IN_BG + text + (" " * pad) + _RESET



class ChatBar:
    """Chat scrolls on top. Footer is two fixed rows: input, then status."""

    def __init__(self):
        # DECSTBM scrolling and cursor-save sequences do not behave reliably
        # across macOS Terminal profiles. A failed repaint can cover every
        # response with the footer's background, which is worse than having no
        # sticky status line. Keep the conventional, dependable terminal
        # prompt by default; allow the experimental bar explicitly.
        self.enabled = bool(
            os.environ.get("LIGHTLX_STICKY_BAR") == "1"
            and sys.stdout.isatty()
            and sys.stdin.isatty()
        )
        self._on = False
        self.sess = None
        self.busy = False
        self._lock = threading.Lock()
        self._prev_winch = None
        self._input = ""
        self._menu_h = 0

    def attach(self, sess):
        self.sess = sess

    def size(self):
        sz = shutil.get_terminal_size((80, 24))
        return max(20, sz.columns), max(8, sz.lines)

    def start(self):
        if not self.enabled:
            return
        self._on = True
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        self._prev_winch = signal.getsignal(signal.SIGWINCH)
        try:
            signal.signal(signal.SIGWINCH, self._winch)
        except Exception:
            self._prev_winch = None
        self._set_scroll(0)
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
        _, rows = self.size()
        sys.stdout.write("\033[r\033[0m\033[?25h")
        sys.stdout.write(f"\033[{rows - 1};1H\033[2K\033[{rows};1H\033[2K\n")
        sys.stdout.flush()

    def _set_scroll(self, menu_h=0):
        _, rows = self.size()
        bottom, _, _, _ = footer_rows(rows, menu_h)
        sys.stdout.write(f"\033[1;{bottom}r")
        sys.stdout.flush()

    def _winch(self, signum, frame):
        if self._on:
            self._set_scroll(self._menu_h)
            self.paint()
        prev = self._prev_winch
        if callable(prev):
            prev(signum, frame)

    def _park_chat(self):
        _, rows = self.size()
        bottom, _, _, _ = footer_rows(rows, self._menu_h)
        sys.stdout.write(f"\033[{bottom};1H")
        sys.stdout.flush()

    def paint(self):
        if not self._on or self.sess is None:
            return
        with self._lock:
            cols, rows = self.size()
            _, _, input_row, status_row = footer_rows(rows, self._menu_h)
            inp = format_input_line(self._input, cols, busy=self.busy)
            bar = format_bar(self.sess, cols, busy=self.busy, color=True)
            sys.stdout.write(f"\033[{input_row};1H\033[2K{inp}")
            sys.stdout.write(f"\033[{status_row};1H\033[2K{bar}")
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
        self._input = ""
        self._menu_h = 0
        self._set_scroll(0)
        self.paint()
        if os.name != "posix" or not sys.stdin.isatty():
            cols, rows = self.size()
            _, _, input_row, _ = footer_rows(rows, 0)
            sys.stdout.write(f"\033[{input_row};1H\033[2K\033[0m")
            sys.stdout.flush()
            return input("\033[1m›\033[0m ")
        return self._readline_tty()

    def _readline_tty(self) -> str:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        buf = ""
        sel = 0
        try:
            tty.setraw(fd)
            sys.stdout.write("\033[?25h")
            self._render_footer(buf, [], 0)
            while True:
                ch = sys.stdin.read(1)
                if not ch:
                    raise EOFError
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\x04":
                    if not buf:
                        raise EOFError
                    continue
                if ch in ("\r", "\n"):
                    catalog = slash_catalog(self.sess)
                    buf, submit = slash_enter(buf, catalog, sel)
                    if not submit:
                        sel = 0
                        self._render_footer(buf, filter_slash(catalog, buf), sel)
                        continue
                    self._commit_line(buf)
                    return buf
                if ch == "\x1b":
                    nxt = sys.stdin.read(1) if self._stdin_ready() else ""
                    if nxt == "[":
                        code = sys.stdin.read(1)
                        hits = filter_slash(slash_catalog(self.sess), buf)
                        if code == "A" and hits:
                            sel = (sel - 1) % len(hits)
                        elif code == "B" and hits:
                            sel = (sel + 1) % len(hits)
                        self._render_footer(buf, hits, sel)
                    else:
                        buf = ""
                        sel = 0
                        self._render_footer(buf, [], 0)
                    continue
                if ch in ("\x7f", "\x08"):
                    buf = buf[:-1]
                    sel = 0
                elif ch == "\t":
                    hits = filter_slash(slash_catalog(self.sess), buf)
                    if hits:
                        cmd = hits[min(sel, len(hits) - 1)][0]
                        buf = cmd + (" " if cmd in _SLASH_NEEDS_SPACE else "")
                        sel = 0
                elif ch == "\x15":
                    buf = ""
                    sel = 0
                elif ord(ch) >= 32:
                    buf += ch
                    sel = 0
                hits = filter_slash(slash_catalog(self.sess), buf)
                if sel >= len(hits):
                    sel = 0
                self._render_footer(buf, hits, sel)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._input = ""
            self._menu_h = 0
            self._set_scroll(0)
            self.paint()
            self._park_chat()

    def _stdin_ready(self):
        import select
        try:
            return bool(select.select([sys.stdin], [], [], 0.05)[0])
        except Exception:
            return False

    def _render_footer(self, buf, hits, sel):
        cols, rows = self.size()
        n = min(len(hits), MENU_MAX)
        scroll_bottom, menu_top, input_row, status_row = footer_rows(rows, n)
        old_h = self._menu_h
        self._menu_h = n
        self._input = buf
        if old_h > n:
            _, old_top, _, _ = footer_rows(rows, old_h)
            for r in range(old_top, menu_top):
                sys.stdout.write(f"\033[{r};1H\033[2K\033[0m")
        self._set_scroll(n)
        for i, (cmd, hint) in enumerate(hits[:n]):
            mark = "▸" if i == sel else " "
            body = f" {mark} {cmd:<12} {hint}"
            vis = body[: max(1, cols)]
            pad = max(0, cols - len(_ANSI.sub("", vis)))
            bg = _MENU_SEL if i == sel else _MENU_BG
            sys.stdout.write(f"\033[{menu_top + i};1H\033[2K{bg}{vis}{' ' * pad}{_RESET}")
        inp = format_input_line(buf, cols, busy=False)
        bar = format_bar(self.sess, cols, busy=False, color=True) if self.sess else (" " * cols)
        sys.stdout.write(f"\033[{input_row};1H\033[2K{inp}")
        sys.stdout.write(f"\033[{status_row};1H\033[2K{bar}")
        shown = " › " + buf
        caret = 4 + len(buf)
        if len(shown) > cols:
            caret = cols
        sys.stdout.write(f"\033[{input_row};{min(cols, caret)}H\033[?25h")
        sys.stdout.flush()

    def _commit_line(self, buf):
        cols, rows = self.size()
        self._menu_h = 0
        self._input = ""
        scroll_bottom, _, input_row, status_row = footer_rows(rows, 0)
        self._set_scroll(0)
        for r in range(scroll_bottom + 1, input_row):
            sys.stdout.write(f"\033[{r};1H\033[2K\033[0m")
        shown = buf if len(buf) < cols - 2 else buf[: cols - 3] + "…"
        sys.stdout.write(f"\033[{scroll_bottom};1H\033[0m› {shown}\n")
        sys.stdout.write(f"\033[{input_row};1H\033[2K{format_input_line('', cols, busy=True)}")
        if self.sess:
            sys.stdout.write(
                f"\033[{status_row};1H\033[2K{format_bar(self.sess, cols, busy=True, color=True)}"
            )
        sys.stdout.write(f"\033[{scroll_bottom};1H")
        sys.stdout.flush()

    def suspend(self):
        if not self._on:
            return
        _, rows = self.size()
        self._menu_h = 0
        sys.stdout.write("\033[r\033[0m")
        sys.stdout.write(f"\033[{rows - 1};1H\033[2K\033[{rows};1H\033[2K\n")
        sys.stdout.flush()

    def resume(self):
        if not self.enabled or not self._on:
            return
        sys.stdout.write("\n\n")
        self._set_scroll(0)
        self.paint()

    @contextmanager
    def paused(self):
        self.suspend()
        try:
            yield
        finally:
            self.resume()
