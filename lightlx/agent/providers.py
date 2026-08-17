import json
import os
import uuid
import urllib.error
import urllib.request

from .parse import collapse_repeats, is_repeating
from .types import Completion, ToolCall

UA = "lightlx/0.2.0"
DEFAULT_OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if DEFAULT_OLLAMA and "://" not in DEFAULT_OLLAMA:
    DEFAULT_OLLAMA = "http://" + DEFAULT_OLLAMA
DEFAULT_LMSTUDIO = os.environ.get("LM_STUDIO_HOST", "http://127.0.0.1:1234")
if DEFAULT_LMSTUDIO and "://" not in DEFAULT_LMSTUDIO:
    DEFAULT_LMSTUDIO = "http://" + DEFAULT_LMSTUDIO


def _headers(api_key=None):
    h = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": UA}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def http_json(method, url, body=None, headers=None, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    h = {"Accept": "application/json", "User-Agent": UA}
    if data is not None:
        h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return (json.loads(raw) if raw else {}), getattr(r, "status", 200)
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")[:800]
        raise RuntimeError(f"{e.code} {url}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url}: {e.reason}") from e


def probe(url, timeout=2.5, api_key=None):
    h = {"User-Agent": UA, "Accept": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, method="GET", headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            data = json.loads(raw) if raw else {}
            return {"ok": True, "status": getattr(r, "status", 200), "data": data}
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "status": e.code,
            "data": None,
            "auth_required": e.code in (401, 403),
        }
    except Exception:
        return {"ok": False, "status": 0, "data": None}


def probe_ollama(base=None):
    base = (base or DEFAULT_OLLAMA).rstrip("/")
    r = probe(base + "/api/tags")
    if not r.get("ok"):
        return None
    data = r.get("data") or {}
    names = [m.get("name") or m.get("model") for m in data.get("models") or [] if m.get("name") or m.get("model")]
    return {"url": base, "models": names}


def _lmstudio_keys(explicit=None):
    seen = []
    for k in (
        explicit,
        os.environ.get("LM_API_TOKEN"),
        os.environ.get("LM_STUDIO_API_KEY"),
        "lm-studio",
        None,
    ):
        if k not in seen:
            seen.append(k)
    return seen


def _chat_model_ids(rows):
    chat, other = [], []
    for m in rows or []:
        mid = m.get("id") if isinstance(m, dict) else None
        if not mid:
            continue
        blob = " ".join(str(m.get(k) or "") for k in ("id", "type", "object")).lower()
        if "embed" in blob:
            other.append(mid)
        else:
            chat.append(mid)
    return chat + other


def probe_lmstudio(base=None, api_key=None):
    base = (base or DEFAULT_LMSTUDIO).rstrip("/")
    last = None
    for key in _lmstudio_keys(api_key):
        r = probe(base + "/v1/models", api_key=key)
        last = r
        if r.get("ok"):
            names = _chat_model_ids((r.get("data") or {}).get("data"))
            return {"url": base, "models": names, "api_key": key}
        if r.get("status") == 0:
            break
    if last and last.get("auth_required"):
        return {"url": base, "models": [], "auth_required": True}
    fb = _lms_cli_probe()
    if fb:
        fb.setdefault("url", base)
        return fb
    return None


def _lms_cli_probe():
    import shutil
    import subprocess
    lms = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms")
    if not lms or not os.path.isfile(lms):
        return None
    try:
        st = subprocess.run([lms, "status"], capture_output=True, text=True, timeout=4)
    except Exception:
        return None
    text = (st.stdout or "") + "\n" + (st.stderr or "")
    if "Server: ON" not in text and "ON (port" not in text:
        return None
    port = 1234
    for part in text.replace(")", " ").split():
        if part.isdigit() and 1 < int(part) < 65535 and "port" in text.lower():
            port = int(part)
            break
    models = []
    try:
        ps = subprocess.run([lms, "ps"], capture_output=True, text=True, timeout=4)
        for line in (ps.stdout or "").splitlines()[1:]:
            ident = line.split()[0] if line.strip() else ""
            if ident and ident.upper() != "IDENTIFIER":
                models.append(ident)
    except Exception:
        pass
    return {"url": f"http://127.0.0.1:{port}", "models": models, "via": "lms"}


def message_text(val):
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for p in val:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)
    return str(val)


def parse_args(raw):
    if isinstance(raw, dict):
        return raw
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"value": v}
    except json.JSONDecodeError:
        return {"_raw": s}


def _tool_calls_from_openai(message):
    out = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        out.append(ToolCall(
            id=tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
            name=fn.get("name") or "",
            arguments=parse_args(args),
            raw_arguments=args if isinstance(args, str) else json.dumps(args or {}),
        ))
    return out


class StreamAcc:
    def __init__(self):
        self.content = []
        self.reasoning = []
        self.tools = {}
        self.looped = False

    def feed(self, chunk):
        pieces = []
        for c in chunk.get("choices") or []:
            d = c.get("delta") or c.get("message") or {}
            text = message_text(d.get("content"))
            think = message_text(d.get("reasoning_content") or d.get("reasoning") or d.get("thinking"))
            if text:
                self.content.append(text)
                pieces.append(text)
            if think:
                self.reasoning.append(think)
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = self.tools.setdefault(i, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
        blob = "".join(self.content)
        if is_repeating(blob) or is_repeating("".join(self.reasoning)):
            self.looped = True
            pieces = []
        return "".join(pieces)

    def result(self, finish="stop"):
        calls = []
        for i in sorted(self.tools):
            slot = self.tools[i]
            if not slot["name"]:
                continue
            calls.append(ToolCall(
                id=slot["id"] or f"call_{i}_{uuid.uuid4().hex[:8]}",
                name=slot["name"],
                arguments=parse_args(slot["arguments"]),
                raw_arguments=slot["arguments"],
            ))
        text = collapse_repeats("".join(self.content))
        if not text and not calls:
            text = collapse_repeats("".join(self.reasoning))
        return Completion(text, calls, "tool_calls" if calls else finish)


class OpenAICompat:
    kind = "openai"
    parallel_safe = True

    def __init__(self, base_url, model, api_key=None, timeout=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        self.timeout = timeout
        self.label = model
        self.context_length = 0

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.2, on_text=None):
        from .context import sanitize_messages
        messages = sanitize_messages(messages)
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "frequency_penalty": 0.4,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        url = self.base_url + "/v1/chat/completions"
        try:
            return self._stream(url, body, on_text)
        except RuntimeError as e:
            if "404" not in str(e) and "stream" not in str(e).lower():
                raise
            body["stream"] = False
            data, _ = http_json("POST", url, body, _headers(self.api_key), timeout=self.timeout or 600)
            msg = ((data.get("choices") or [{}])[0].get("message")) or {}
            content = message_text(msg.get("content")) or message_text(
                msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking")
            )
            if on_text and content:
                on_text(content)
            calls = _tool_calls_from_openai(msg)
            return Completion(content, calls, "tool_calls" if calls else "stop")

    def _stream(self, url, body, on_text):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={**_headers(self.api_key), "Accept": "text/event-stream"},
        )
        acc = StreamAcc()
        finish = "stop"
        try:
            with urllib.request.urlopen(req, timeout=self.timeout or 600) as resp:
                buf = b""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        s = line.decode("utf-8", "replace").strip()
                        if not s.startswith("data:"):
                            continue
                        payload = s[5:].strip()
                        if payload == "[DONE]":
                            return acc.result(finish)
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        piece = acc.feed(data)
                        if piece and on_text:
                            on_text(piece)
                        if acc.looped:
                            return acc.result("stop")
                        for c in data.get("choices") or []:
                            if c.get("finish_reason"):
                                finish = c["finish_reason"]
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")[:800]
            raise RuntimeError(f"{e.code} {url}: {err}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url}: {e.reason}") from e
        return acc.result(finish)


class Ollama(OpenAICompat):
    kind = "ollama"

    def __init__(self, model, base_url=None, api_key=None, timeout=None):
        super().__init__(base_url or DEFAULT_OLLAMA, model, api_key=api_key or "ollama", timeout=timeout)
        self.label = f"ollama/{model}"

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.2, on_text=None):
        try:
            return super().complete(messages, tools, max_tokens, temperature, on_text)
        except RuntimeError as e:
            if "cannot reach" in str(e) or not (tools or True):
                raise
            return self._native(messages, tools, max_tokens, temperature, on_text)

    def _native(self, messages, tools, max_tokens, temperature, on_text):
        body = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if tools:
            body["tools"] = tools
        url = self.base_url + "/api/chat"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST", headers=_headers(self.api_key),
        )
        content = []
        calls = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout or 600) as resp:
                for raw in resp:
                    try:
                        data = json.loads(raw.decode())
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message") or {}
                    piece = msg.get("content") or ""
                    if piece:
                        content.append(piece)
                        if on_text:
                            on_text(piece)
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            parsed, raw_a = parse_args(args), args
                        else:
                            parsed, raw_a = (args or {}), json.dumps(args or {})
                        calls.append(ToolCall(
                            id=tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
                            name=fn.get("name") or "",
                            arguments=parsed,
                            raw_arguments=raw_a,
                        ))
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")[:800]
            raise RuntimeError(f"{e.code} {url}: {err}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url}: {e.reason}") from e
        text = "".join(content)
        return Completion(text, calls, "tool_calls" if calls else "stop")


class LMStudio(OpenAICompat):
    kind = "lmstudio"

    def __init__(self, model, base_url=None, api_key=None, timeout=None):
        super().__init__(base_url or DEFAULT_LMSTUDIO, model, api_key=api_key or "lm-studio", timeout=timeout)
        self.label = f"lmstudio/{model}"


class CustomOpenAI(OpenAICompat):
    kind = "openai"

    def __init__(self, model, base_url, api_key=None, timeout=None):
        super().__init__(base_url, model, api_key=api_key, timeout=timeout)
        self.label = f"openai/{model}"


def _to_ollama_messages(messages):
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "content": m.get("content") or "",
                "name": m.get("name") or "",
            })
            continue
        item = {"role": role, "content": m.get("content") or ""}
        if m.get("tool_calls"):
            item["tool_calls"] = []
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    item["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    })
                else:
                    item["tool_calls"].append(tc)
        out.append(item)
    return out


class MlxLocal:
    kind = "mlx"
    parallel_safe = False

    def __init__(self, generate_fn, model, tok, eos, think=False, ctx_limit=8192, name="mlx"):
        self._generate = generate_fn
        self.model = model
        self.tok = tok
        self.eos = eos
        self.think = think
        self.ctx_limit = ctx_limit
        self.label = name
        self.verbose = True
        self.context_length = ctx_limit

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.2, on_text=None):
        from .parse import parse_text_tool_calls
        flat = flatten_for_chat(messages)
        buf = []

        def emit(t):
            buf.append(t)
            if on_text:
                on_text(t)

        text = self._generate(
            self.model, self.tok, self.eos, flat, max_tokens,
            verbose=False, think=self.think, ctx_limit=self.ctx_limit, on_token=emit,
        )
        if not buf and text and on_text:
            on_text(text)
        content, calls = parse_text_tool_calls(text or "".join(buf))
        return Completion(content, calls, "tool_calls" if calls else "stop")


def flatten_for_chat(messages):
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = m.get("name") or ""
            out.append({
                "role": "user",
                "content": f"<tool_result name=\"{name}\">\n{m.get('content') or ''}\n</tool_result>",
            })
        elif role == "assistant" and m.get("tool_calls"):
            content = m.get("content") or ""
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    payload = {"name": tc.name, "arguments": tc.arguments}
                else:
                    fn = (tc.get("function") or {})
                    payload = {"name": fn.get("name"), "arguments": parse_args(fn.get("arguments"))}
                content += "\n<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def to_openai_messages(messages):
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            item = {"role": "assistant", "content": m.get("content") or None, "tool_calls": []}
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    item["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)},
                    })
                else:
                    item["tool_calls"].append(tc)
            out.append(item)
        elif m.get("role") == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or m.get("id") or "",
                "name": m.get("name") or "",
                "content": m.get("content") or "",
            })
        else:
            out.append({"role": m.get("role"), "content": m.get("content") or ""})
    return out
