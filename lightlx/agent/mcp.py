import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

from .types import ToolSpec

PROTOCOL = "2024-11-05"
CONFIG_PATH = os.path.expanduser("~/.lightlx/mcp.json")


def load_mcp_config(workspace=None, extra=None):
    servers = {}
    for path in (
        CONFIG_PATH,
        str(Path(workspace or ".") / ".lightlx" / "mcp.json"),
        str(Path(workspace or ".") / "mcp.json"),
        extra,
    ):
        if not path or not os.path.isfile(path):
            continue
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        block = data.get("mcpServers") or data.get("servers") or {}
        if isinstance(block, dict):
            servers.update(block)
    return servers


class MCPServer:
    def __init__(self, name, spec, workspace="."):
        self.name = name
        self.spec = spec
        self.workspace = workspace
        self.proc = None
        self._id = 0
        self._lock = threading.Lock()
        self._pending = {}
        self._reader = None
        self._stderr_thread = None
        self._stderr_tail = ""
        self.tools = []
        self.info = {}
        self.error = None

    def start(self, timeout=20):
        if self.spec.get("disabled"):
            raise RuntimeError("disabled")
        cmd = self.spec.get("command")
        args = list(self.spec.get("args") or [])
        if not cmd:
            url = self.spec.get("url")
            if url:
                raise RuntimeError("HTTP MCP is not built in — wrap it with a stdio bridge")
            raise RuntimeError("missing command")
        env = os.environ.copy()
        env.update({k: str(v) for k, v in (self.spec.get("env") or {}).items()})
        cwd = self.spec.get("cwd") or self.workspace
        self.proc = subprocess.Popen(
            [cmd, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            bufsize=0,
        )
        self._stderr_tail = ""
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.proc.stderr,), daemon=True,
        )
        self._stderr_thread.start()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "lightlx", "version": "0.2.1"},
        }, timeout=timeout)
        self.info = result or {}
        self.notify("notifications/initialized", {})
        listed = self.request("tools/list", {}, timeout=timeout) or {}
        self.tools = listed.get("tools") or []

    def close(self):
        if not self.proc:
            return
        proc = self.proc
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
        self.proc = None

    def _drain_stderr(self, err):
        if not err:
            return
        try:
            while True:
                chunk = err.read(4096)
                if not chunk:
                    return
                text = chunk.decode("utf-8", "replace")
                self._stderr_tail = (self._stderr_tail + text)[-8000:]
        except Exception:
            return

    def _next_id(self):
        with self._lock:
            self._id += 1
            return self._id

    def request(self, method, params, timeout=60):
        mid = self._next_id()
        ev = threading.Event()
        box = {}
        self._pending[mid] = (ev, box)
        self._write({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        if not ev.wait(timeout):
            self._pending.pop(mid, None)
            raise TimeoutError(f"{self.name} {method} timed out")
        if "error" in box:
            err = box["error"]
            raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))
        return box.get("result")

    def notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def call_tool(self, name, arguments, timeout=120):
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        return _render_mcp_result(result)

    def read_resource(self, uri, timeout=60):
        result = self.request("resources/read", {"uri": uri}, timeout=timeout)
        return _render_mcp_result(result)

    def _write(self, msg):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError(f"{self.name} is not running")
        data = json.dumps(msg).encode()
        header = f"Content-Length: {len(data)}\r\n\r\n".encode()
        with self._lock:
            self.proc.stdin.write(header + data)
            self.proc.stdin.flush()

    def _read_loop(self):
        stdout = self.proc.stdout
        while self.proc and stdout:
            msg = _read_framed(stdout)
            if msg is None:
                return
            if "id" in msg and ("result" in msg or "error" in msg):
                pending = self._pending.pop(msg["id"], None)
                if pending:
                    ev, box = pending
                    if "error" in msg:
                        box["error"] = msg["error"]
                    else:
                        box["result"] = msg.get("result")
                    ev.set()


def _read_framed(stream):
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            if headers:
                break
            continue
        if line[:1] == b"{":
            try:
                return json.loads(line.decode())
            except json.JSONDecodeError:
                return None
        if b":" in line:
            k, v = line.decode("utf-8", "replace").split(":", 1)
            headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length") or 0)
    if n <= 0:
        return None
    body = b""
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            return None
        body += chunk
    try:
        return json.loads(body.decode())
    except json.JSONDecodeError:
        return None


def _render_mcp_result(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    parts = []
    is_err = bool(result.get("isError")) if isinstance(result, dict) else False
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(item.get("text") or "")
            elif item.get("type") == "resource":
                res = item.get("resource") or {}
                parts.append(res.get("text") or res.get("uri") or json.dumps(res))
            else:
                parts.append(json.dumps(item))
    elif isinstance(result, dict):
        parts.append(json.dumps(result, indent=2)[:20000])
    text = "\n".join(p for p in parts if p)
    if is_err:
        return "MCP error:\n" + text
    return text


def _slug(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)[:40]


def _tool_name(server, name):
    n = f"mcp_{_slug(server)}_{_slug(name)}"
    if len(n) <= 64:
        return n
    return n[:55] + "_" + uuid.uuid4().hex[:8]


class MCPHub:
    def __init__(self):
        self.servers = {}
        self.specs = []

    def connect(self, config, workspace=".", on_status=None):
        for name, spec in (config or {}).items():
            if not isinstance(spec, dict) or spec.get("disabled"):
                continue
            srv = MCPServer(name, spec, workspace)
            try:
                srv.start()
                self.servers[name] = srv
                if on_status:
                    on_status(name, True, f"{len(srv.tools)} tools")
            except Exception as e:
                srv.close()
                if on_status:
                    on_status(name, False, str(e))

    def close(self):
        for srv in self.servers.values():
            srv.close()
        self.servers.clear()

    def tool_specs(self):
        specs = []
        for sname, srv in self.servers.items():
            for t in srv.tools:
                tname = t.get("name") or "tool"
                full = _tool_name(sname, tname)
                desc = t.get("description") or f"MCP {sname}/{tname}"
                params = t.get("inputSchema") or {"type": "object", "properties": {}}

                def _make(server=srv, original=tname):
                    def handler(**kwargs):
                        return server.call_tool(original, kwargs)
                    return handler

                specs.append(ToolSpec(
                    name=full,
                    description=f"[MCP:{sname}] {desc}",
                    parameters=params,
                    handler=_make(),
                    source=f"mcp:{sname}",
                ))
        return specs

    def summary(self):
        lines = []
        for name, srv in self.servers.items():
            lines.append(f"{name}: {len(srv.tools)} tools")
        return lines
