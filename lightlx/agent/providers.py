import json
import os
import re
import threading
import time
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
        err = _clean_http_error(e.read().decode(errors="replace"))
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
    loaded = []
    ps = probe(base + "/api/ps")
    if ps.get("ok"):
        for m in ((ps.get("data") or {}).get("models") or []):
            n = m.get("name") or m.get("model")
            if n:
                loaded.append(n)
    return {"url": base, "models": loaded or names, "listed": names, "loaded": loaded}


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


def split_lmstudio_models(rows):
    """Return (loaded_chat, listed_chat) from LM Studio /api/v0/models rows."""
    loaded, listed, _ = parse_lmstudio_rows(rows)
    return loaded, listed


def parse_lmstudio_rows(rows):
    loaded, listed, details = [], [], {}
    for m in rows or []:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("name")
        if not mid:
            continue
        blob = " ".join(str(m.get(k) or "") for k in ("id", "type", "object")).lower()
        if "embed" in blob:
            continue
        listed.append(mid)
        state = str(m.get("state") or "").lower()
        is_loaded = state in ("loaded", "loading", "idle") or m.get("loaded") is True
        if is_loaded:
            loaded.append(mid)
        details[mid] = {
            "capabilities": (
                [str(c).lower() for c in m["capabilities"]]
                if isinstance(m.get("capabilities"), list) else None
            ),
            "context": m.get("loaded_context_length") or m.get("max_context_length"),
            "state": state,
            "type": m.get("type") or "",
        }
    return loaded, listed, details


def caps_from_list(raw):
    caps = [str(c).lower().replace("-", "_") for c in (raw or [])]
    tools = any(
        c in ("tool_use", "tools", "tool", "function_calling", "functions")
        for c in caps
    )
    return {"tools": tools, "raw": caps}


def infer_caps(details, model_id, context_length=0):
    d = (details or {}).get(model_id) or {}
    raw = d.get("capabilities")
    if raw is None:
        info = {"tools": True, "raw": []}
    else:
        info = caps_from_list(raw)
    ctx = int(d.get("context") or context_length or 0)
    info["context"] = ctx
    # Subagents need headroom for a fresh system+task prompt alongside the
    # parent context; 4096 is enough. Avoid hardcoding a large 8192 gate.
    info["subagents"] = bool(info["tools"] and (not ctx or ctx >= 4096))
    return info


def runtime_notice(old, new):
    """Human lines for a model/capability change. old/new are dicts with model, caps."""
    if not new:
        return []
    lines = []
    caps = new.get("caps") or {}
    old_caps = (old or {}).get("caps") or {}
    model_changed = bool(
        old and old.get("model") and new.get("model") and old["model"] != new["model"]
    )
    if model_changed:
        lines.append(f"loaded model is now {new['model']}")
    tools = bool(caps.get("tools"))
    subs = bool(caps.get("subagents"))
    old_tools = bool(old_caps.get("tools")) if old else None
    old_subs = bool(old_caps.get("subagents")) if old else None
    if not tools:
        if old is None or old_tools is not False:
            lines.append(
                "this model does not advertise tool calling — "
                "using text tool format, running inline (no subagents)"
            )
    elif not subs:
        if old is None or old_subs:
            lines.append("this model can use tools but not nested subagents — running inline")
    elif old is not None and (not old_tools or not old_subs):
        lines.append("tools + subagents enabled")
    return lines


def probe_lmstudio(base=None, api_key=None):
    base = (base or DEFAULT_LMSTUDIO).rstrip("/")
    last = None
    for key in _lmstudio_keys(api_key):
        r0 = probe(base + "/api/v0/models", api_key=key)
        last = r0
        if r0.get("ok"):
            rows = (r0.get("data") or {}).get("data") or (r0.get("data") or {}).get("models") or []
            loaded, listed, details = parse_lmstudio_rows(rows)
            if not listed:
                r = probe(base + "/v1/models", api_key=key)
                listed = _chat_model_ids((r.get("data") or {}).get("data")) if r.get("ok") else []
            return {
                "url": base,
                "models": loaded or listed,
                "loaded": loaded,
                "listed": listed,
                "details": details,
                "api_key": key,
            }
        r = probe(base + "/v1/models", api_key=key)
        last = r
        if r.get("ok"):
            names = _chat_model_ids((r.get("data") or {}).get("data"))
            fb = _lms_cli_probe() or {}
            loaded = list(fb.get("models") or [])
            loaded = [n for n in loaded if n in names] or loaded
            return {
                "url": base,
                "models": loaded or names,
                "loaded": loaded,
                "listed": names,
                "api_key": key,
            }
        if r.get("status") == 0 and r0.get("status") == 0:
            break
    if last and last.get("auth_required"):
        return {"url": base, "models": [], "loaded": [], "auth_required": True}
    fb = _lms_cli_probe()
    if fb:
        fb.setdefault("url", base)
        fb.setdefault("loaded", list(fb.get("models") or []))
        fb.setdefault("listed", list(fb.get("models") or []))
        return fb
    return None


def ollama_model_caps(base, model):
    if not base or not model:
        return {}
    try:
        data, _ = http_json("POST", base.rstrip("/") + "/api/show", {"name": model}, timeout=8)
    except Exception:
        return {}
    caps = data.get("capabilities") or []
    info = caps_from_list(caps)
    if not info["tools"]:
        tmpl = str(data.get("template") or "") + " " + str(data.get("modelfile") or "")
        info["tools"] = "tool" in tmpl.lower() or ".Tools" in tmpl
    return info


def refresh_remote_provider(provider):
    """Follow the currently loaded model and return {model, caps} or None."""
    kind = getattr(provider, "kind", "")
    if kind not in ("lmstudio", "ollama"):
        ctx = int(getattr(provider, "context_length", 0) or 0)
        tools = True
        return {
            "model": getattr(provider, "model", None) or getattr(provider, "label", ""),
            "caps": {"tools": tools, "subagents": bool(not ctx or ctx >= 4096), "raw": [], "context": ctx},
        }
    if kind == "lmstudio":
        info = probe_lmstudio(getattr(provider, "base_url", None), getattr(provider, "api_key", None))
        if not info:
            return None
        loaded = list(info.get("loaded") or [])
        current = getattr(provider, "model", "")
        model = current if current in loaded else (loaded[0] if loaded else current)
        if model and model != current:
            provider.model = model
            provider.label = f"lmstudio/{model}"
        details = info.get("details") or {}
        ctx = 0
        d = details.get(model) or {}
        try:
            ctx = int(d.get("context") or 0)
        except (TypeError, ValueError):
            ctx = 0
        if ctx:
            provider.context_length = ctx
        caps = infer_caps(details, model, ctx)
        provider.capabilities = caps
        return {"model": model, "caps": caps}
    info = probe_ollama(getattr(provider, "base_url", None))
    if not info:
        return None
    loaded = list(info.get("loaded") or [])
    current = getattr(provider, "model", "")
    model = current if current in loaded else (loaded[0] if loaded else current)
    if model and model != current:
        provider.model = model
        provider.label = f"ollama/{model}"
    ocap = ollama_model_caps(info.get("url") or getattr(provider, "base_url", ""), model)
    ctx = int(getattr(provider, "context_length", 0) or 0)
    caps = {
        "tools": ocap.get("tools", True) if ocap else True,
        "raw": (ocap or {}).get("raw") or [],
        "context": ctx,
        "subagents": bool((ocap.get("tools", True) if ocap else True) and (not ctx or ctx >= 4096)),
    }
    provider.capabilities = caps
    return {"model": model, "caps": caps}


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
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        args = fn.get("arguments") if isinstance(fn, dict) else None
        name = (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name") or ""
        out.append(ToolCall(
            id=tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
            name=name or "",
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
                args = fn.get("arguments")
                if args is not None:
                    if isinstance(args, str):
                        slot["arguments"] += args
                    elif isinstance(args, dict):
                        # Some servers send arguments as object per chunk; merge
                        existing = parse_args(slot["arguments"])
                        existing.update(args)
                        slot["arguments"] = json.dumps(existing)
        blob = "".join(self.content)
        # Only abort on repeating *answer* tokens. Qwen-style reasoning often
        # loops "wait/hmm" and aborting that closes the LM Studio socket before
        # any plan text is produced.
        if is_repeating(blob):
            self.looped = True
            pieces = []
        return "".join(pieces)

    def result(self, finish="stop"):
        calls = []
        for i in sorted(self.tools):
            slot = self.tools[i]
            # Keep nameless slots that carry arguments: dropping them silently
            # turns a malformed stream into an empty completion; an empty name
            # surfaces as a visible "unknown tool" error the model can retry.
            if not slot["name"] and not (slot["arguments"] or "").strip():
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
    _SLOT_COOLDOWN = 0.0

    def __init__(self, base_url, model, api_key=None, timeout=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"
        self.timeout = timeout
        self.label = model
        self.context_length = 0
        self.frequency_penalty = 0.4
        self._slot_until = 0.0

    def _wait_slot(self):
        gap = self._slot_until - time.time()
        if gap > 0:
            time.sleep(gap)

    def _cooldown(self, seconds):
        self._slot_until = max(self._slot_until, time.time() + seconds)

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.2, on_text=None):
        from .context import normalize_messages, sanitize_messages
        messages = sanitize_messages(normalize_messages(messages))
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        url = self.base_url + "/v1/chat/completions"

        def send(payload):
            try:
                return self._stream(url, payload, on_text)
            except RuntimeError as e:
                if "404" not in str(e) and "stream" not in str(e).lower():
                    raise
                payload = dict(payload)
                payload["stream"] = False
                data, _ = http_json("POST", url, payload, _headers(self.api_key), timeout=self.timeout or 600)
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = message_text(msg.get("content")) or message_text(
                    msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking")
                )
                if on_text and content:
                    on_text(content)
                calls = _tool_calls_from_openai(msg)
                finish = choice.get("finish_reason") or "stop"
                return Completion(content, calls, "tool_calls" if calls else finish)

        serial = not getattr(self, "parallel_safe", True)
        attempts = 5 if serial else 1
        backoff = (3.0, 6.0, 10.0, 15.0) if serial else (1.0, 2.0)
        self._wait_slot()
        try:
            for i in range(attempts):
                try:
                    out = send(body)
                    if serial and getattr(out, "finish", "") == "disconnected":
                        self._cooldown(4.0)
                    else:
                        self._cooldown(self._SLOT_COOLDOWN or 1.0)
                    return out
                except RuntimeError as e:
                    code = _http_code(e)
                    if code == "400" and "frequency_penalty" in body:
                        body = dict(body)
                        body.pop("frequency_penalty", None)
                        return send(body)
                    if code in _TRANSIENT_HTTP and i + 1 < attempts:
                        wait = backoff[min(i, len(backoff) - 1)]
                        self._cooldown(wait)
                        self._wait_slot()
                        continue
                    if serial and code in _TRANSIENT_HTTP:
                        self._cooldown(10.0)
                    raise
        finally:
            if serial:
                self._cooldown(self._SLOT_COOLDOWN or 1.0)

    def _stream(self, url, body, on_text):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={**_headers(self.api_key), "Accept": "text/event-stream"},
        )
        acc = StreamAcc()
        finish = "stop"
        saw_done = False
        aborted = False
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
                            saw_done = True
                            return acc.result(finish)
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        piece = acc.feed(data)
                        if piece and on_text:
                            on_text(piece)
                        if acc.looped:
                            aborted = True
                            break
                        for c in data.get("choices") or []:
                            if c.get("finish_reason"):
                                finish = c["finish_reason"]
                    if aborted:
                        # Drain the socket so LM Studio can finish cleanly
                        # before LightLX opens the next request.
                        while resp.read(4096):
                            pass
                        break
        except urllib.error.HTTPError as e:
            err = _clean_http_error(e.read().decode(errors="replace"))
            if not getattr(self, "parallel_safe", True):
                self._cooldown(6.0)
            raise RuntimeError(f"{e.code} {url}: {err}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url}: {e.reason}") from e
        if aborted:
            self._cooldown(4.0)
            return acc.result("stop")
        if not saw_done and finish == "stop":
            finish = "disconnected"
            if not getattr(self, "parallel_safe", True):
                self._cooldown(4.0)
        return acc.result(finish)


_TRANSIENT_HTTP = {"408", "409", "425", "429", "500", "502", "503", "504"}


def _clean_http_error(err):
    # LM Studio serves bare HTML error pages on 500s; the real message lives in
    # the <pre> body. Prefer that, else strip all tags to a single readable line.
    t = (err or "")[:800]
    m = re.search(r"<pre[^>]*>(.*?)</pre>", t, re.I | re.S)
    if m:
        t = m.group(1)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:160] or "HTTP error"


def _http_code(err):
    m = re.match(r"\s*(\d{3})\b", str(err))
    return m.group(1) if m else ""


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
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        name = (fn.get("name") if isinstance(fn, dict) else None) or tc.get("name") or ""
                        if isinstance(args, str):
                            parsed, raw_a = parse_args(args), args
                        else:
                            parsed, raw_a = (args or {}), json.dumps(args or {})
                        calls.append(ToolCall(
                            id=tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
                            name=name or "",
                            arguments=parsed,
                            raw_arguments=raw_a,
                        ))
        except urllib.error.HTTPError as e:
            err = _clean_http_error(e.read().decode(errors="replace"))
            raise RuntimeError(f"{e.code} {url}: {err}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {url}: {e.reason}") from e
        text = "".join(content)
        return Completion(text, calls, "tool_calls" if calls else "stop")


class LMStudio(OpenAICompat):
    kind = "lmstudio"
    # One loaded model cannot serve several chat streams at once. Parallel
    # `task` calls make LM Studio drop the parent connection ("Client
    # disconnected") and LightLX ends the turn with an empty plan.
    parallel_safe = False
    _SLOT_COOLDOWN = 0.75
    _SLOT = threading.Lock()

    def __init__(self, model, base_url=None, api_key=None, timeout=None):
        super().__init__(base_url or DEFAULT_LMSTUDIO, model, api_key=api_key or "lm-studio", timeout=timeout)
        self.label = f"lmstudio/{model}"
        self.frequency_penalty = None

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.2, on_text=None):
        with self._SLOT:
            return super().complete(messages, tools, max_tokens, temperature, on_text)


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
                "content": message_text(m.get("content")),
                "name": m.get("name") or "",
            })
            continue
        item = {"role": role, "content": message_text(m.get("content"))}
        if m.get("tool_calls"):
            item["tool_calls"] = []
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    item["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    })
                elif isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    item["tool_calls"].append({
                        "id": tc.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": fn.get("name") or tc.get("name") or "",
                            "arguments": fn.get("arguments") or tc.get("arguments") or {},
                        },
                    })
                else:
                    item["tool_calls"].append(tc)
        out.append(item)
    return out


class MlxLocal:
    kind = "mlx"
    parallel_safe = False

    def __init__(self, generate_fn, model, tok, eos, think=False, ctx_limit=32768, name="mlx"):
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
            content = message_text(m.get("content"))
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    payload = {"name": tc.name, "arguments": tc.arguments}
                elif isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    payload = {
                        "name": fn.get("name") or tc.get("name") or "",
                        "arguments": fn.get("arguments") or tc.get("arguments") or {},
                    }
                else:
                    fn = (tc.get("function") or {})
                    payload = {"name": fn.get("name"), "arguments": parse_args(fn.get("arguments"))}
                content += "\n<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": message_text(m.get("content"))})
    return out


def to_openai_messages(messages):
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            item = {"role": "assistant", "content": message_text(m.get("content")) or None, "tool_calls": []}
            for tc in m["tool_calls"]:
                if isinstance(tc, ToolCall):
                    raw = tc.raw_arguments
                    if raw and isinstance(raw, str):
                        try:
                            json.loads(raw)
                        except Exception:
                            raw = json.dumps(tc.arguments if isinstance(tc.arguments, dict) else {})
                    else:
                        raw = json.dumps(tc.arguments if isinstance(tc.arguments, dict) else {})
                    item["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": raw},
                    })
                elif isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or ""
                    args = fn.get("arguments") or tc.get("arguments") or tc.get("raw_arguments") or ""
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    elif isinstance(args, str):
                        try:
                            json.loads(args)
                        except Exception:
                            args = json.dumps({"command": args} if name == "bash" else {"value": args})
                    else:
                        args = "{}"
                    item["tool_calls"].append({
                        "id": tc.get("id") or "",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                else:
                    item["tool_calls"].append(tc)
            out.append(item)
        elif m.get("role") == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or m.get("id") or "",
                "name": m.get("name") or "",
                "content": message_text(m.get("content")),
            })
        else:
            out.append({"role": m.get("role"), "content": message_text(m.get("content"))})
    return out
