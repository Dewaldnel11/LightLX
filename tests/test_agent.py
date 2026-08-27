import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from lightlx.agent.parse import parse_text_tool_calls
from lightlx.agent.prompts import DOC_ALIASES, format_tool_list, system_prompt
from lightlx.agent.providers import StreamAcc, parse_args
from lightlx.agent.tools import BuiltinTools, Workspace, html_to_text, summarize_call
from lightlx.agent.types import ToolSpec


class RepeatTests(unittest.TestCase):
    def test_collapse_and_detect(self):
        from lightlx.agent.parse import collapse_repeats, is_repeating, looks_like_tool_narration
        looped = "Let me read more files.\n\n" * 8
        self.assertTrue(is_repeating(looped))
        self.assertEqual(collapse_repeats(looped), "Let me read more files.")
        self.assertTrue(looks_like_tool_narration("Let me read the scan_bridge.py"))
        self.assertFalse(looks_like_tool_narration("Here is the improvement plan:\n1. tests\n2. UI"))
        self.assertTrue(looks_like_tool_narration("Now let me add the audit log querying functionality:"))
        self.assertTrue(looks_like_tool_narration("Next, I'll implement the retry logic in ScannerBridge."))
        self.assertTrue(looks_like_tool_narration("1. records.py\n2. audit.py\n3. policy.py"))
        self.assertTrue(looks_like_tool_narration("A subagent is working on the fix"))
        self.assertTrue(looks_like_tool_narration("I am awaiting reports from these subagents"))
        self.assertTrue(looks_like_tool_narration("The implementation is currently underway via parallel subagents"))
        self.assertTrue(looks_like_tool_narration("I have applied edits to the scan bridge"))

    def test_completion_text_signals(self):
        from lightlx.agent.parse import completion_text_signals, is_implementation_request
        yaml = "name: YouGuard\n" + ("key: value\n" * 120)
        blob = f"```yaml\n{yaml}\n```\nNow I'll write `.github/workflows/ci.yml`."
        self.assertGreater(len(blob), 1200)
        sig = completion_text_signals(blob)
        self.assertTrue(sig["unapplied_code"])
        self.assertTrue(sig["action_promise"])
        self.assertIn(".github/workflows/ci.yml", sig["announced_paths"])
        done = completion_text_signals("Implemented the requested fix.")
        self.assertTrue(done["done_claim"])
        tool = completion_text_signals(
            '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}\n</tool_call>'
        )
        self.assertFalse(tool["unapplied_code"])
        self.assertTrue(is_implementation_request("Write the files"))
        self.assertTrue(is_implementation_request("implement the P0 items"))
        self.assertTrue(is_implementation_request("Please finish implementing the previous kickoff work"))
        self.assertTrue(is_implementation_request("keep writing the swift files"))
        self.assertTrue(is_implementation_request("fixing the scan bridge now"))
        self.assertFalse(is_implementation_request("Why does the agent stop after narration?"))
        self.assertFalse(is_implementation_request("Follow the kickoff protocol. Topic:\nfoo"))


class ParseTests(unittest.TestCase):
    def test_json_block(self):
        text = 'hello\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>\n'
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(content, "hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "a.py")

    def test_qwen_block(self):
        text = "<tool_call>\nbash\n<arg_key>command</arg_key>\n<arg_value>ls -la</arg_value>\n</tool_call>"
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(calls[0].arguments["command"], "ls -la")
        self.assertFalse(content)

    def test_fence(self):
        text = '```tool_call\n{"name": "glob", "arguments": {"pattern": "**/*.py"}}\n```'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(calls[0].name, "glob")
        self.assertEqual(calls[0].arguments["pattern"], "**/*.py")

    def test_no_tools(self):
        content, calls = parse_text_tool_calls("just a reply")
        self.assertEqual(content, "just a reply")
        self.assertEqual(calls, [])

    def test_tool_code_json_fence(self):
        text = '```tool_code\n{"name": "read_file", "parameters": {"path": "a.py"}}\n```'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "a.py")

    def test_tool_code_python(self):
        text = '```tool_code\nprint(default_api.read_file(path="src/app.py", offset=1))\n```'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "src/app.py")
        self.assertEqual(calls[0].arguments["offset"], 1)

    def test_function_xml(self):
        text = '<function=bash>{"command": "ls -la"}</function>'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(calls[0].arguments["command"], "ls -la")

    def test_invoke_line(self):
        text = 'invoke tool read_file with path is a.py'
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "a.py")

    def test_mangled_kv_write_file(self):
        # Weak local models emit a tool call as bare text inside a ```bash
        # fence with stray closing tags. It must still be recovered so the
        # file actually gets written.
        text = (
            "Now let me write this file:\n\n"
            "```bash\n"
            "write_file path=/tmp/quantme/.github/workflows/build.yml "
            "content=name: Build\n\n"
            "on:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  build:\n    runs-on: macos-latest\n"
            "</parameter>\n</function>\n</tool_call>\n"
            "```\n"
        )
        content, calls = parse_text_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "write_file")
        self.assertEqual(
            calls[0].arguments["path"],
            "/tmp/quantme/.github/workflows/build.yml",
        )
        body = calls[0].arguments["content"]
        self.assertTrue(body.startswith("name: Build"))
        self.assertIn("runs-on: macos-latest", body)
        self.assertNotIn("</parameter>", body)
        self.assertNotIn("</tool_call>", body)
        self.assertNotIn("write_file", content)

    def test_mangled_kv_bare_read(self):
        _, calls = parse_text_tool_calls("read_file path=lightlx/agent/loop.py")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].arguments["path"], "lightlx/agent/loop.py")

    def test_mangled_kv_prose_does_not_misfire(self):
        _, calls = parse_text_tool_calls(
            "You can call write_file to save the config later."
        )
        self.assertEqual(calls, [])

    def test_fenced_bash_shell_call(self):
        text = (
            "```bash\n"
            "find /tmp/quantme -name KICKOFF_PLAN.md -type f 2>/dev/null\n"
            "```"
        )
        _, calls = parse_text_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertIn("KICKOFF_PLAN.md", calls[0].arguments["command"])

    def test_resume_kickoff_is_implementation(self):
        from lightlx.agent.parse import is_implementation_request
        self.assertTrue(is_implementation_request(
            "Please resume the previous work that the kickoff plan had layed out"
        ))


class ProviderErrorTests(unittest.TestCase):
    def test_clean_http_error_strips_html(self):
        from lightlx.agent.providers import _clean_http_error

        html = (
            '<!DOCTYPE html><html><head><title>Error</title></head>'
            '<body><pre>Internal Server Error</pre></body></html>'
        )
        out = _clean_http_error(html)
        self.assertEqual(out, "Internal Server Error")
        self.assertNotIn("<", out)

    def test_http_code_parses(self):
        from lightlx.agent.providers import _http_code

        self.assertEqual(_http_code(RuntimeError("500 http://x: Internal Server Error")), "500")
        self.assertEqual(_http_code(RuntimeError("cannot reach x")), "")


class StreamGateTests(unittest.TestCase):
    def _run(self, text, char_by_char=False):
        from lightlx.agent.parse import StreamGate

        emitted = []
        hits = {"n": 0}
        gate = StreamGate(emitted.append, lambda: hits.__setitem__("n", hits["n"] + 1))
        if char_by_char:
            for ch in text:
                gate.feed(ch)
        else:
            gate.feed(text)
        gate.close()
        return "".join(emitted), gate.suppressed, hits["n"]

    def test_prose_streams(self):
        out, suppressed, hits = self._run("Here is the plan:\n1. do a thing\n")
        self.assertFalse(suppressed)
        self.assertEqual(hits, 0)
        self.assertIn("Here is the plan:", out)

    def test_genuine_code_fence_streams(self):
        text = "See this:\n```python\nprint('hello world here')\n```\n"
        out, suppressed, hits = self._run(text)
        self.assertFalse(suppressed)
        self.assertIn("print('hello world here')", out)

    def test_fenced_tool_call_suppressed(self):
        text = (
            "Now let me write this file:\n\n"
            "```bash\n"
            "write_file path=/tmp/a.yml content=name: Build\n"
            "on:\n  push:\n</tool_call>\n```\n"
        )
        out, suppressed, hits = self._run(text)
        self.assertTrue(suppressed)
        self.assertEqual(hits, 1)
        self.assertIn("Now let me write this file:", out)
        self.assertNotIn("write_file", out)
        self.assertNotIn("name: Build", out)

    def test_fenced_tool_call_suppressed_char_stream(self):
        text = (
            "```bash\nwrite_file path=/tmp/a.yml content=name: Build\nmore\n```\n"
        )
        out, suppressed, hits = self._run(text, char_by_char=True)
        self.assertTrue(suppressed)
        self.assertNotIn("write_file", out)

    def test_shell_fence_suppressed(self):
        text = (
            "```bash\n"
            "find /tmp -name KICKOFF_PLAN.md 2>/dev/null\n"
            "```\n"
        )
        out, suppressed, hits = self._run(text)
        self.assertTrue(suppressed)
        self.assertNotIn("find", out)
        self.assertEqual(hits, 1)

    def test_bare_tool_call_suppressed(self):
        out, suppressed, hits = self._run("write_file path=x.txt content=hi\n")
        self.assertTrue(suppressed)
        self.assertEqual(out, "")

    def test_tool_call_xml_suppressed(self):
        out, suppressed, hits = self._run("<tool_call>\nbash\n</tool_call>\n")
        self.assertTrue(suppressed)


class StreamAccTests(unittest.TestCase):
    def test_content_and_tools(self):
        acc = StreamAcc()
        acc.feed({"choices": [{"delta": {"content": "Hi"}}]})
        acc.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": ""}}
        ]}}]})
        acc.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"path": "x"}'}}
        ]}}]})
        result = acc.result()
        self.assertEqual(result.content, "Hi")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments["path"], "x")

    def test_nameless_slot_kept(self):
        acc = StreamAcc()
        acc.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"arguments": '{"path": "x"}'}}
        ]}}]})
        result = acc.result()
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "")
        self.assertEqual(result.tool_calls[0].arguments["path"], "x")

    def test_dict_args_merge(self):
        acc = StreamAcc()
        acc.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": {"path": "a"}}}
        ]}}]})
        acc.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": {"offset": 2}}}
        ]}}]})
        result = acc.result()
        self.assertEqual(result.tool_calls[0].arguments["path"], "a")
        self.assertEqual(result.tool_calls[0].arguments["offset"], 2)


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hi')\nprint('bye')\n")
        (self.root / "README.md").write_text("# hello\n")
        self.tools = BuiltinTools(Workspace(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_write_edit(self):
        self.assertIn("print('hi')", self.tools.read_file("src/app.py"))
        self.tools.write_file("src/new.py", "x = 1\n")
        self.assertTrue((self.root / "src" / "new.py").is_file())
        out = self.tools.edit_file("src/new.py", "x = 1", "x = 2")
        self.assertIn("edited", out)
        self.assertEqual((self.root / "src" / "new.py").read_text(), "x = 2\n")

    def test_glob_grep_list(self):
        hits = self.tools.glob("**/*.py")
        self.assertIn("src/app.py", hits)
        grepped = self.tools.grep("print", path="src")
        self.assertIn("app.py", grepped)
        listing = self.tools.list_dir(".")
        self.assertIn("src/", listing)

    def test_edit_not_unique(self):
        (self.root / "dup.txt").write_text("aa\naa\n")
        out = self.tools.edit_file("dup.txt", "aa", "bb")
        self.assertIn("error", out)
        out = self.tools.edit_file("dup.txt", "aa", "bb", replace_all=True)
        self.assertIn("2 replacement", out)

    def test_html_to_text(self):
        text = html_to_text("<html><script>x</script><p>Hello <b>world</b></p></html>")
        self.assertIn("Hello", text)
        self.assertNotIn("x", text)

    def test_summarize(self):
        self.assertEqual(summarize_call("bash", {"command": "ls"}), "bash  ls")
        longp = "/Users/dewaldnel/Library/Mobile Documents/com~apple~CloudDocs/NelCapital Stuff/Experimental/9.0/app.py"
        s = summarize_call("read_file", {"path": longp})
        self.assertNotIn("Mobile Documents", s)
        self.assertIn("app.py", s)

    def test_docs_alias(self):
        self.assertEqual(DOC_ALIASES["claude-code"], "anthropics/claude-code")
        self.assertEqual(DOC_ALIASES["codex"], "openai/codex")

    def test_unknown_docs_source(self):
        out = self.tools.docs("not-a-real-alias")
        self.assertIn("unknown source", out)

    def test_missing_args_error(self):
        from lightlx.agent.loop import _run_one
        spec = self.tools._spec("edit_file", "d", {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }, self.tools.edit_file)
        out = _run_one(spec, {})
        self.assertIn("missing required argument", out)
        self.assertIn("path", out)


class PromptTests(unittest.TestCase):
    def test_system_mentions_workspace(self):
        spec = ToolSpec("read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, lambda **k: "")
        text = system_prompt("/tmp/ws", tools_as_text=True, tool_list=format_tool_list([spec]))
        self.assertIn("/tmp/ws", text)
        self.assertIn("read_file", text)
        self.assertIn("task subagents", text)

    def test_system_omits_task_when_no_subagents(self):
        text = system_prompt("/tmp/ws", subagents=False)
        self.assertIn("cannot run nested agents", text)
        self.assertNotIn("task needs description", text)


class ParseArgsTests(unittest.TestCase):
    def test_parse_args(self):
        self.assertEqual(parse_args('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_args({"a": 1}), {"a": 1})
        self.assertEqual(parse_args(""), {})

    def test_to_openai_normalizes_content(self):
        from lightlx.agent.providers import to_openai_messages
        msgs = [
            {"role": "user", "content": {"text": "hi"}},
            {"role": "assistant", "content": ["part1", "part2"]},
            {"role": "tool", "content": {"err": "x"}, "tool_call_id": "c1"},
        ]
        out = to_openai_messages(msgs)
        for m in out:
            self.assertIsInstance(m.get("content"), (str, type(None)))

    def test_to_openai_dict_tool_calls(self):
        from lightlx.agent.providers import to_openai_messages
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "name": "edit_file",
                "arguments": {"path": "a.py", "old_string": "x", "new_string": "y"},
            }],
        }]
        out = to_openai_messages(msgs)
        tc = out[0]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "edit_file")
        self.assertIsInstance(tc["function"]["arguments"], str)
        self.assertIn("a.py", tc["function"]["arguments"])


class ContextTests(unittest.TestCase):
    def test_estimate_and_threshold(self):
        from lightlx.agent.context import estimate_tokens, handoff_note, needs_compact
        msgs = [{"role": "user", "content": "x" * 4000}]
        self.assertGreater(estimate_tokens(msgs), 800)
        self.assertTrue(needs_compact(msgs, context_length=1024, max_tokens=256))
        self.assertFalse(needs_compact([{"role": "user", "content": "hi"}], 8192, 512))
        note = handoff_note("ollama/a", "lmstudio/b", 32768)
        self.assertIn("handoff", note["content"])
        self.assertIn("32768", note["content"])

    def test_completion_tokens_uses_remaining_window(self):
        from lightlx.agent.context import completion_tokens, room_for
        msgs = [{"role": "user", "content": "hi"}]
        n = completion_tokens(msgs, context_length=50432, cap=0)
        self.assertGreater(n, 40_000)
        n_cap = completion_tokens(msgs, context_length=50432, cap=2048)
        self.assertEqual(n_cap, 2048)
        # auto cap must leave compact headroom
        self.assertGreater(room_for(50432, 0), 10_000)
        self.assertGreater(room_for(50432, 50432), 10_000)

    def test_full_window_not_clamped(self):
        # Large local windows should be used for replies, not clamped to a
        # small fixed cap.
        from lightlx.agent.context import completion_tokens, room_for
        msgs = [{"role": "user", "content": "hi"}]
        big = completion_tokens(msgs, context_length=131072, cap=0)
        self.assertGreater(big, 100_000)
        # room_for scales with the window rather than capping at a fixed 2048
        self.assertGreater(room_for(262144, 0), 190_000)

    def test_mlx_default_ctx_limit_lifted(self):
        import inspect
        from lightlx.agent.providers import MlxLocal
        sig = inspect.signature(MlxLocal.__init__)
        default = sig.parameters["ctx_limit"].default
        self.assertGreaterEqual(default, 32768)

    def test_sanitize_drops_orphan_tools(self):
        from lightlx.agent.context import normalize_messages, sanitize_messages
        from lightlx.agent.types import ToolCall
        tc = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "nope"},
            {"role": "assistant", "content": "", "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "ok"},
        ]
        out = sanitize_messages(msgs)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "tool"])
        self.assertEqual(out[-1]["content"], "ok")

    def test_normalize_merges_empty_assistants(self):
        from lightlx.agent.context import normalize_messages

        raw = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "part one"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "part two"},
            {"role": "user", "content": "again"},
        ]
        out = normalize_messages(raw)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1]["content"], "part one\n\npart two")

    def test_compact_keeps_recent(self):
        from lightlx.agent.context import compact_messages
        from lightlx.agent.types import Completion

        class Fake:
            def complete(self, messages, **kwargs):
                return Completion("goals: ship it")

        hist = []
        for i in range(10):
            hist.append({"role": "user", "content": f"u{i} " + "n" * 20})
            hist.append({"role": "assistant", "content": f"a{i}"})
        out, did = compact_messages(Fake(), hist, keep=4)
        self.assertTrue(did)
        self.assertIn("compacted", out[0]["content"])
        self.assertTrue(any(m.get("content") == "a9" for m in out))
        self.assertIn("Continue", out[-1]["content"])


class ResumeTests(unittest.TestCase):
    def test_hydrate_and_unanswered(self):
        from lightlx.agent.loop import unanswered_tool_calls
        from lightlx.agent.sessions import hydrate_messages
        from lightlx.agent.types import ToolCall
        raw = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}},
                {"id": "c2", "name": "read_file", "arguments": {"path": "b.py"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "ok"},
        ]
        msgs = hydrate_messages(raw)
        self.assertIsInstance(msgs[1]["tool_calls"][0], ToolCall)
        left = unanswered_tool_calls(msgs)
        self.assertEqual(len(left), 1)
        self.assertEqual(left[0].name, "read_file")
        self.assertEqual(left[0].arguments["path"], "b.py")


class SessionStoreTests(unittest.TestCase):
    def test_save_load(self):
        from lightlx.agent import sessions
        old = sessions.SESS_DIR
        tmp = tempfile.TemporaryDirectory()
        sessions.SESS_DIR = tmp.name
        try:
            sid = sessions.save_session({
                "title": "hello world",
                "history": [{"role": "user", "content": "hello world"}],
                "provider": "ollama/qwen",
                "source": {"kind": "ollama", "model": "qwen"},
            })
            rec = sessions.load_session(sid)
            self.assertEqual(rec["title"], "hello world")
            self.assertEqual(sessions.list_sessions(5)[0]["id"], sid)
            self.assertTrue(sessions.delete_session(sid))
        finally:
            sessions.SESS_DIR = old
            tmp.cleanup()

    def test_one_resume_per_project(self):
        from lightlx.agent import sessions
        old = sessions.SESS_DIR
        tmp = tempfile.TemporaryDirectory()
        sessions.SESS_DIR = tmp.name
        try:
            a = sessions.save_session({
                "title": "old", "workspace": "/tmp/proj",
                "history": [{"role": "user", "content": "old"}],
            })
            b = sessions.save_session({
                "title": "new", "workspace": "/tmp/proj",
                "history": [{"role": "user", "content": "new"}],
            })
            self.assertNotEqual(a, b)
            rows = sessions.list_sessions(10, one_per_project=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], b)
            self.assertIsNone(sessions.load_session(a))
            other = sessions.save_session({
                "title": "other", "workspace": "/tmp/other",
                "history": [{"role": "user", "content": "x"}],
            })
            rows = sessions.list_sessions(10, one_per_project=True)
            ids = {r["id"] for r in rows}
            self.assertEqual(ids, {b, other})
            self.assertEqual(sessions.project_name({"workspace": "/tmp/proj"}), "proj")
        finally:
            sessions.SESS_DIR = old
            tmp.cleanup()


class LoadedModelTests(unittest.TestCase):
    def test_split_lmstudio_models(self):
        from lightlx.agent.providers import split_lmstudio_models
        rows = [
            {"id": "qwen/qwen3.5-9b", "type": "vlm", "state": "loaded"},
            {"id": "google/gemma-4-e4b", "type": "vlm", "state": "not-loaded"},
            {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "not-loaded"},
        ]
        loaded, listed = split_lmstudio_models(rows)
        self.assertEqual(loaded, ["qwen/qwen3.5-9b"])
        self.assertEqual(listed, ["qwen/qwen3.5-9b", "google/gemma-4-e4b"])

    def test_infer_caps_tools_and_subagents(self):
        from lightlx.agent.providers import infer_caps, runtime_notice
        details = {
            "qwen": {"capabilities": ["tool_use"], "context": 50432},
            "tiny": {"capabilities": ["tool_use"], "context": 2048},
            "plain": {"capabilities": [], "context": 32768},
            "unknown": {"capabilities": None, "context": 32768},
        }
        q = infer_caps(details, "qwen")
        self.assertTrue(q["tools"])
        self.assertTrue(q["subagents"])
        t = infer_caps(details, "tiny")
        self.assertTrue(t["tools"])
        self.assertFalse(t["subagents"])
        p = infer_caps(details, "plain")
        self.assertFalse(p["tools"])
        self.assertFalse(p["subagents"])
        u = infer_caps(details, "unknown")
        self.assertTrue(u["tools"])
        self.assertTrue(u["subagents"])
        first = runtime_notice(None, {"model": "qwen", "caps": q})
        self.assertEqual(first, [])
        inline = runtime_notice(None, {"model": "tiny", "caps": t})
        self.assertTrue(any("inline" in line for line in inline))
        switch = runtime_notice(
            {"model": "tiny", "caps": t},
            {"model": "qwen", "caps": q},
        )
        self.assertTrue(any("qwen" in line for line in switch))
        self.assertTrue(any("subagents enabled" in line for line in switch))
        same = runtime_notice({"model": "tiny", "caps": t}, {"model": "tiny", "caps": t})
        self.assertEqual(same, [])


class SkillsMemoryTests(unittest.TestCase):
    def test_frontmatter_and_discover(self):
        from lightlx.agent.memory import split_frontmatter
        from lightlx.agent.skills import discover_skills, expand_skill
        meta, body = split_frontmatter("---\ndescription: hello\npaths:\n  - \"*.py\"\n---\n# Hi\n")
        self.assertEqual(meta["description"], "hello")
        self.assertIn("*.py", meta["paths"])
        self.assertIn("# Hi", body)
        skills = discover_skills(tempfile.gettempdir())
        self.assertIn("code-review", skills)
        text = expand_skill(skills["summarize-changes"], workspace=tempfile.gettempdir())
        self.assertIn("Skill: summarize-changes", text)
        from lightlx.agent.skills import import_skills
        tmp = tempfile.TemporaryDirectory()
        src = Path(tmp.name) / "fake-skill"
        src.mkdir()
        (src / "SKILL.md").write_text("---\ndescription: imported\ncontext: fork\n---\nDo the thing\n")
        from lightlx.agent.skills import _load_skill_md
        sk = _load_skill_md("fake-skill", src / "SKILL.md", "claude-user")
        self.assertTrue(sk.fork)
        dest = Path(tmp.name) / "dest"
        reports = import_skills([sk], dest)
        self.assertTrue((dest / "fake-skill" / "SKILL.md").is_file())
        self.assertTrue(any("copied" in r for r in reports))
        tmp.cleanup()

    def test_instructions_and_imports(self):
        from lightlx.agent.memory import format_instructions, load_instructions
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "README.md").write_text("# Demo\n")
        (root / "LIGHTLX.md").write_text("Use 2-space indent.\nSee @README.md\n")
        blocks = load_instructions(root)
        blob = format_instructions(blocks)
        self.assertIn("2-space", blob)
        self.assertIn("Demo", blob)
        tmp.cleanup()

    def test_memory_store(self):
        from lightlx.agent import memory as memmod
        old = memmod.MEMORY_ROOT
        tmp = tempfile.TemporaryDirectory()
        memmod.MEMORY_ROOT = Path(tmp.name)
        try:
            store = memmod.MemoryStore(tmp.name)
            store.write("MEMORY.md", "# Memory\n- uses pnpm\n")
            store.write("debugging.md", "check the cache\n")
            self.assertIn("pnpm", store.load_index())
            self.assertIn("debugging.md", store.list_files())
            self.assertIn("cache", store.read("debugging.md"))
            extra = store.format_for_prompt()
            self.assertIn("pnpm", extra)
        finally:
            memmod.MEMORY_ROOT = old
            tmp.cleanup()

    def test_init_writes_once(self):
        from lightlx.agent.memory import init_project
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "pyproject.toml").write_text("[project]\nname='x'\n")
        msg = init_project(root)
        self.assertTrue((root / "LIGHTLX.md").is_file())
        self.assertIn("wrote", msg)
        again = init_project(root)
        self.assertIn("already exists", again)
        tmp.cleanup()


class CompatLoopTests(unittest.TestCase):
    def test_text_mode_parses_and_runs(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolSpec

        calls = []

        def read_file(path):
            calls.append(path)
            return f"contents of {path}"

        registry = {
            "read_file": ToolSpec(
                "read_file", "Read",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                read_file,
            ),
        }

        class Fake:
            parallel_safe = True

            def __init__(self):
                self.n = 0

            def complete(self, messages, tools=None, **kwargs):
                self.n += 1
                if self.n == 1:
                    return Completion(
                        '```tool_code\nprint(read_file(path="a.py"))\n```',
                        [],
                        "stop",
                    )
                return Completion("done", [], "stop")

        result = run_agent(
            Fake(), [{"role": "user", "content": "read it"}], registry,
            native_tools=False, max_iters=5,
        )
        self.assertEqual(calls, ["a.py"])
        self.assertEqual(result.status, "done")
        self.assertEqual(result.text, "done")

    def test_alias_and_arg_remap(self):
        from lightlx.agent.loop import execute_tools
        from lightlx.agent.types import ToolCall, ToolSpec

        seen = {}

        def bash(command):
            seen["command"] = command
            return "ok"

        registry = {
            "bash": ToolSpec(
                "bash", "Run",
                {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                bash,
            ),
        }
        outs = execute_tools(
            [ToolCall(id="c1", name="run_terminal_cmd", arguments={"cmd": "ls"})],
            registry, parallel=False,
        )
        self.assertEqual(outs, ["ok"])
        self.assertEqual(seen["command"], "ls")

    def test_git_snapshot(self):
        from lightlx.agent.loop import _git_snapshot
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        if os.system(f"git -C {root} init -q") != 0:
            tmp.cleanup()
            self.skipTest("git missing")
        (root / "a.txt").write_text("hi\n")
        snap = _git_snapshot(str(root))
        self.assertIsNotNone(snap)
        status, _ = snap
        self.assertIn("a.txt", status)
        tmp.cleanup()

    def test_agent_status_max_iters(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolSpec

        class Looping:
            parallel_safe = True

            def complete(self, messages, **kwargs):
                return Completion("", [], "stop")

        registry = {
            "read_file": ToolSpec("read_file", "r", {"type": "object", "properties": {}}, lambda **k: ""),
        }
        result = run_agent(
            Looping(), [{"role": "user", "content": "x"}], registry,
            native_tools=False, max_iters=2,
        )
        self.assertEqual(result.status, "empty")

    def test_empty_then_plan(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolSpec

        class OnceEmpty:
            parallel_safe = True

            def __init__(self):
                self.n = 0

            def complete(self, messages, **kwargs):
                self.n += 1
                if self.n == 1:
                    return Completion("", [], "disconnected")
                return Completion("# Plan\n1. map repo", [], "stop")

        registry = {
            "read_file": ToolSpec("read_file", "r", {"type": "object", "properties": {}}, lambda **k: ""),
        }
        result = run_agent(
            OnceEmpty(), [{"role": "user", "content": "kickoff"}], registry,
            native_tools=False, max_iters=5,
        )
        self.assertEqual(result.status, "done")
        self.assertIn("Plan", result.text)


class CompletionLoopTests(unittest.TestCase):
    def _registry(self):
        from lightlx.agent.types import ToolSpec
        written = {}

        def write_file(path, content=""):
            written[path] = content
            return "ok"

        def grep(pattern="", path="."):
            return "no matches"

        def edit_file(path, old_string="", new_string=""):
            written[path] = new_string
            return "ok"

        registry = {
            "write_file": ToolSpec(
                "write_file", "w",
                {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                 "required": ["path", "content"]},
                write_file,
            ),
            "edit_file": ToolSpec(
                "edit_file", "e",
                {"type": "object", "properties": {
                    "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
                }, "required": ["path", "old_string", "new_string"]},
                edit_file,
            ),
            "grep": ToolSpec(
                "grep", "g",
                {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
                grep,
            ),
        }
        return registry, written

    def test_long_narration_then_write(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolCall
        registry, written = self._registry()
        events = []
        yaml = "name: YouGuard\n" + ("key: value\n" * 120)
        dump = f"```yaml\n{yaml}\n```\nNow I'll write `.github/workflows/ci.yml`."

        class Fake:
            parallel_safe = False

            def __init__(self):
                self.n = 0

            def complete(self, messages, tools=None, **kwargs):
                self.n += 1
                if self.n == 1:
                    return Completion(dump, [], "stop")
                if self.n == 2:
                    return Completion("", [ToolCall(
                        id="w1", name="write_file",
                        arguments={"path": ".github/workflows/ci.yml", "content": "ok"},
                    )], "tool_calls")
                return Completion("Implemented CI workflow.", [], "stop")

        fake = Fake()
        result = run_agent(
            fake, [{"role": "user", "content": "Write the CI workflow files"}],
            registry, native_tools=True, max_iters=10,
            on_event=lambda kind, **kw: events.append((kind, kw)),
            completion_mode="implementation",
        )
        checks = [e for e in events if e[0] == "completion_check"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(written[".github/workflows/ci.yml"], "ok")
        self.assertEqual(result.status, "done")
        self.assertEqual(fake.n, 3)

    def test_search_only_false_done(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolCall
        registry, written = self._registry()
        events = []

        class Fake:
            parallel_safe = False

            def __init__(self):
                self.n = 0

            def complete(self, messages, tools=None, **kwargs):
                self.n += 1
                if self.n == 1:
                    return Completion("", [ToolCall(
                        id="g1", name="grep", arguments={"pattern": "embedding"},
                    )], "tool_calls")
                return Completion("Done.", [], "stop")

        fake = Fake()
        result = run_agent(
            fake, [{"role": "user", "content": "implement persist embeddings"}],
            registry, native_tools=True, max_iters=20,
            on_event=lambda kind, **kw: events.append((kind, kw)),
            completion_mode="implementation",
        )
        checks = [e for e in events if e[0] == "completion_check"]
        self.assertEqual(len(checks), 6)
        self.assertFalse(written)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(fake.n, 8)

    def test_qa_not_forced(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion
        registry, _ = self._registry()
        events = []

        class Fake:
            parallel_safe = True

            def complete(self, messages, **kwargs):
                return Completion("run_agent ends a turn when a completion has no tool calls.", [], "stop")

        result = run_agent(
            Fake(), [{"role": "user", "content": "Why does the agent stop after narration?"}],
            registry, native_tools=False, max_iters=5,
            on_event=lambda kind, **kw: events.append((kind, kw)),
            completion_mode="auto",
        )
        self.assertEqual(result.status, "done")
        self.assertFalse(any(e[0] == "completion_check" for e in events))

    def test_kickoff_plan_not_forced(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion
        registry, written = self._registry()
        events = []

        class Fake:
            parallel_safe = True

            def complete(self, messages, **kwargs):
                return Completion("## Plan\n1. map repo\n2. add tests", [], "stop")

        result = run_agent(
            Fake(), [{"role": "user", "content": "Follow the kickoff protocol. Topic:\nfoo"}],
            registry, native_tools=False, max_iters=5,
            on_event=lambda kind, **kw: events.append((kind, kw)),
            completion_mode="plan",
        )
        self.assertEqual(result.status, "done")
        self.assertFalse(any(e[0] == "completion_check" for e in events))
        self.assertFalse(written)

    def test_lmstudio_serializes_tools(self):
        from lightlx.agent.providers import LMStudio, StreamAcc
        self.assertFalse(LMStudio("qwen").parallel_safe)
        acc = StreamAcc()
        think = "Wait. " * 40
        acc.feed({"choices": [{"delta": {"reasoning_content": think}}]})
        self.assertFalse(acc.looped)


class SessionPersistTests(unittest.TestCase):
    def test_record_from_keeps_task_reports(self):
        from lightlx.agent.sessions import record_from
        from lightlx.agent.types import ToolCall

        class FakeSess:
            session_id = "s1"
            history = [
                {"role": "user", "content": "fix auth"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        ToolCall(id="t1", name="task", arguments={"description": "fix", "prompt": "do it"}),
                        ToolCall(id="t2", name="read_file", arguments={"path": "a.py"}),
                    ],
                },
                {"role": "tool", "tool_call_id": "t1", "name": "task", "content": "subagent [implement] fix — done · 3 steps\nchanged a.py"},
                {"role": "tool", "tool_call_id": "t2", "name": "read_file", "content": "x" * 200},
                {"role": "assistant", "content": "all good"},
            ]
            pending = None
            ws = type("W", (), {"root": "/tmp"})()
            provider = type("P", (), {"label": "lmstudio/gemma", "kind": "lmstudio"})()
            max_tokens = 4096

        rec = record_from(FakeSess(), {"kind": "lmstudio"})
        hist = rec["history"]
        roles = [m["role"] for m in hist]
        self.assertIn("tool", roles)
        task = next(m for m in hist if m.get("role") == "tool" and m.get("name") == "task")
        self.assertIn("subagent", task["content"])
        read = next(m for m in hist if m.get("role") == "tool" and m.get("name") == "read_file")
        self.assertEqual(read["content"], "(elided)")
        orphan_asst = next(
            m for m in hist
            if m.get("role") == "assistant" and m.get("content") == "all good"
        )
        self.assertFalse(orphan_asst.get("tool_calls"))


class ChatBarTests(unittest.TestCase):
    def test_fmt_and_bar(self):
        from lightlx.agent.ui import ctx_usage, fmt_tok, format_bar, meter, short_model
        self.assertEqual(fmt_tok(512), "512")
        self.assertEqual(fmt_tok(1500), "1.5k")
        self.assertEqual(fmt_tok(12000), "12k")
        self.assertEqual(meter(0, 8), "░░░░░░░░")
        self.assertEqual(meter(100, 8), "████████")
        self.assertEqual(len(meter(50, 8)), 8)

        class Sess:
            history = [{"role": "user", "content": "x" * 400}]
            context_length = 8192
            native_tools = True
            max_tokens = 4096
            provider = type("P", (), {"label": "lmstudio/qwen/qwen3.5-9b"})()
            ws = type("W", (), {"root": "/tmp/LightLX"})()
            last_turn = {"steps": 4, "dur": "12s"}
            max_tokens = 0

        sess = Sess()
        self.assertEqual(short_model(sess), "qwen3.5-9b")
        used, ctx, pct = ctx_usage(sess)
        self.assertEqual(ctx, 8192)
        self.assertGreater(used, 0)
        self.assertGreaterEqual(pct, 0)
        bar = format_bar(sess, cols=80, color=False)
        self.assertIn("qwen3.5-9b", bar)
        self.assertIn("native", bar)
        self.assertIn("last 4", bar)
        self.assertIn("LightLX", bar)
        self.assertEqual(len(bar), 80)
        busy = format_bar(sess, cols=60, busy=True, color=False)
        self.assertIn("working", busy)

        from lightlx.agent.ui import filter_slash, footer_rows, format_input_line
        sb, mt, ir, sr = footer_rows(32, 0)
        self.assertEqual((ir, sr), (31, 32))
        self.assertEqual(sb, 30)
        sb2, mt2, ir2, sr2 = footer_rows(32, 3)
        self.assertEqual((ir2, sr2), (31, 32))
        self.assertLess(mt2, ir2)
        self.assertLess(sb2, mt2)
        raw = __import__("re").sub(r"\033\[[0-9;]*m", "", format_input_line("hello", 40))
        self.assertEqual(len(raw), 40)
        self.assertIn("›", raw)
        self.assertEqual(filter_slash([("/kickoff", "x"), ("/brain", "y")], "/kickoff How"), [])

    def test_slash_enter_submits_exact_brain(self):
        from lightlx.agent.ui import SLASH_COMMANDS, slash_enter
        buf, submit = slash_enter("/brain", SLASH_COMMANDS)
        self.assertTrue(submit)
        self.assertEqual(buf, "/brain")
        buf, submit = slash_enter("/br", SLASH_COMMANDS)
        self.assertFalse(submit)
        self.assertEqual(buf, "/brain ")
        buf, submit = slash_enter("/help", SLASH_COMMANDS)
        self.assertTrue(submit)
        self.assertEqual(buf, "/help")
        buf, submit = slash_enter("/he", SLASH_COMMANDS)
        self.assertTrue(submit)
        self.assertEqual(buf, "/help")

    def test_sticky_bar_is_opt_in(self):
        from lightlx.agent.ui import ChatBar
        old = os.environ.pop("LIGHTLX_STICKY_BAR", None)
        try:
            self.assertFalse(ChatBar().enabled)
        finally:
            if old is not None:
                os.environ["LIGHTLX_STICKY_BAR"] = old

    def test_slash_filter(self):
        from lightlx.agent.ui import SLASH_COMMANDS, filter_slash, slash_catalog
        all_open = filter_slash(SLASH_COMMANDS, "/")
        self.assertTrue(all_open)
        self.assertTrue(any(c == "/kickoff" for c, _ in SLASH_COMMANDS))
        hits = filter_slash(SLASH_COMMANDS, "/br")
        self.assertEqual(hits[0][0], "/brain")
        self.assertEqual(filter_slash(SLASH_COMMANDS, "hello"), [])

        class Sk:
            name = "code-review"
            user_invocable = True
            description = "Review the diff"

        class Sess:
            skills = {"code-review": Sk()}

        cat = slash_catalog(Sess())
        self.assertTrue(any(c == "/code-review" for c, _ in cat))


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "brain"
        import lightlx.agent.brain as brain
        self.brain = brain
        self._old = brain.BRAIN_ROOT
        brain.BRAIN_ROOT = self.root

    def tearDown(self):
        self.brain.BRAIN_ROOT = self._old
        self.tmp.cleanup()

    def test_redact(self):
        from lightlx.agent.brain import redact
        t = redact("mail a@b.com token=sk-abc123456789 key AKIAIOSFODNN7EXAMPLE")
        self.assertNotIn("a@b.com", t)
        self.assertNotIn("sk-abc", t)
        self.assertIn("[redacted", t)

    def test_digest_skips_low_idle(self):
        from lightlx.agent.brain import digest_for_prompt, write_record
        write_record("correction", "use pnpm not npm", source="user", confidence="high", root=self.root)
        write_record("gotcha", "invented api", source="idle-extract", confidence="low", root=self.root)
        d = digest_for_prompt(self.root)
        self.assertIn("pnpm", d)
        self.assertNotIn("invented api", d)

    def test_search_and_jobs(self):
        from lightlx.agent.brain import brain_search, claim, complete, enqueue, init_jobs, write_record
        write_record("preference", "2-space indent", source="user", root=self.root)
        self.assertIn("2-space", brain_search("2-space", root=self.root))
        init_jobs(self.root)
        enqueue("s1", "/tmp/x.json", root=self.root)
        job = claim(now=1e12, root=self.root)
        self.assertEqual(job["session_id"], "s1")
        complete("s1", root=self.root)
        self.assertIsNone(claim(now=1e12, root=self.root))

    def test_skip_short_session(self):
        from lightlx.agent.brain import should_extract_session
        rec = {"history": [{"role": "user", "content": "hi"}], "updated": "2000-01-01T00:00:00Z"}
        self.assertFalse(should_extract_session(rec, idle_seconds=1, min_turns=4))
        rec["history"] = [{"role": "user", "content": "x"}] * 5
        p = Path(self.tmp.name) / "sess.json"
        p.write_text(json.dumps(rec))
        os.utime(p, (0, 0))
        self.assertTrue(should_extract_session(p, idle_seconds=1, now=1e12, min_turns=4))

    def test_tick_disabled(self):
        from lightlx.agent.brain import tick_idle
        self.assertEqual(tick_idle({"brain.enabled": False}, busy=False, root=self.root), "")

    def test_ddg_parse_and_claims(self):
        from lightlx.agent.brain import claims_missing_sources, parse_ddg_html
        html = '<a class="result__a" href="https://example.com/docs">Official Docs</a>'
        rows = parse_ddg_html(html)
        self.assertEqual(rows[0]["url"], "https://example.com/docs")
        missing = claims_missing_sources("- the widget API uses /v9/frobnicate\n- see https://example.com/v9")
        self.assertTrue(any("frobnicate" in x for x in missing))
        self.assertFalse(any("example.com/v9" in x for x in missing))

    def test_path_scoped_rules(self):
        from lightlx.agent.memory import load_instructions
        ws = Path(self.tmp.name) / "proj"
        rules = ws / ".lightlx" / "rules"
        rules.mkdir(parents=True)
        (rules / "py.md").write_text("---\npaths:\n  - \"*.py\"\n---\nUse ruff.\n")
        (rules / "always.md").write_text("Always be kind.\n")
        always = "\n".join(b for _, b in load_instructions(ws))
        self.assertIn("kind", always)
        self.assertNotIn("ruff", always)
        matched = "\n".join(b for _, b in load_instructions(ws, touched_paths=["src/app.py"]))
        self.assertIn("ruff", matched)

    def test_kickoff_skill_exists(self):
        from lightlx.agent.skills import discover_skills
        skills = discover_skills(self.tmp.name)
        self.assertIn("kickoff", skills)
        self.assertIn("Claims table", skills["kickoff"].body)
        self.assertIn("Do not start implementing", skills["kickoff"].body)

    def test_to_openai_messages_raw_arguments_valid_json(self):
        from lightlx.agent.types import ToolCall
        from lightlx.agent.providers import to_openai_messages
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall("call_1", "bash", {"command": "echo hi"}, raw_arguments="echo hi"),
            ],
        }]
        out = to_openai_messages(msgs)
        self.assertEqual(len(out), 1)
        raw_args = out[0]["tool_calls"][0]["function"]["arguments"]
        # Must be parseable valid JSON string
        parsed = json.loads(raw_args)
        self.assertEqual(parsed, {"command": "echo hi"})

    def test_remap_args_strips_null_like_values(self):
        from lightlx.agent.loop import remap_args
        from lightlx.agent.types import ToolSpec
        spec = ToolSpec("bash", "b", {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"],
        }, lambda **k: "")
        remapped = remap_args(spec, {"command": "null", "timeout": 60})
        self.assertNotIn("command", remapped)
        self.assertEqual(remapped.get("timeout"), 60)

    def test_summarize_call_ignores_null_values(self):
        from lightlx.agent.tools import summarize_call
        self.assertEqual(summarize_call("bash", {"command": "null"}), "bash")
        self.assertEqual(summarize_call("bash", {"command": "ls -la"}), "bash  ls -la")

    def test_bad_tool_call_openai_tool_role_matching(self):
        from lightlx.agent.loop import run_agent
        from lightlx.agent.types import Completion, ToolCall, ToolSpec

        class BadFirstProvider:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools=None, max_tokens=256, temperature=0.2, on_text=None):
                self.calls += 1
                if self.calls == 1:
                    return Completion("", [ToolCall("call_bad", "bash", {})], "tool_calls")
                # In second call, ensure all tool_call_ids have matching tool message
                tool_ids = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
                assert "call_bad" in tool_ids
                return Completion("I have fixed the tool call.", [], "stop")

        spec = ToolSpec("bash", "b", {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }, lambda **k: "ok")
        prov = BadFirstProvider()
        res = run_agent(prov, [{"role": "user", "content": "run ls"}], {"bash": spec}, native_tools=True)
        self.assertEqual(res.status, "done")


class ResumePickerTests(unittest.TestCase):
    def test_pick_resume_labels_with_project_name(self):
        from unittest.mock import patch
        from lightlx.agent import repl
        rows = [{
            "workspace": "/tmp/myproj",
            "provider": "ollama/qwen",
            "updated": "2026-01-01T00:00:00Z",
        }]
        with patch("lightlx.agent.repl.list_sessions", return_value=rows), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            rec = repl._pick_resume()
        self.assertIs(rec, rows[0])


class MlxHandoffTests(unittest.TestCase):
    def test_run_agent_copies_chat_history_and_label(self):
        from unittest.mock import patch
        from lightlx import cli

        class ChatSess:
            history = [
                {"role": "user", "content": "capital of France"},
                {"role": "assistant", "content": "Paris"},
            ]
            name = "GLM-5.2"
            label = "GLM-5.2"
            session_id = "chat-1"
            provider = None

        captured = {}

        class FakeAgent:
            def __init__(self, *a, **k):
                self.history = []
                self.session_id = None
                self.provider = type("P", (), {"label": "ollama/qwen"})()
                self.context_length = 8192

            def apply_handoff(self, old):
                captured["old"] = old
                return False

            def persist(self):
                captured["history"] = list(self.history)

            def close(self):
                pass

        def fake_repl(sess):
            captured["seeded"] = list(sess.history)
            return None

        with patch("lightlx.agent.repl.AgentSession", FakeAgent), \
             patch("lightlx.agent.repl.agent_repl", fake_repl), \
             patch("lightlx.cli.save_state"):
            cli._run_agent(
                type("P", (), {"label": "ollama/qwen"})(),
                {"prefs": {}},
                carry=ChatSess(),
                workspace="/tmp",
            )
        self.assertEqual(captured["old"], "GLM-5.2")
        self.assertEqual(captured["seeded"][0]["content"], "capital of France")
        self.assertEqual(len(captured["seeded"]), 2)

    def test_session_label_follows_name(self):
        from lightlx.cli import Session
        s = Session({"prefs": {}})
        s.name = "GLM-5.2"
        self.assertEqual(s.label, "GLM-5.2")


class MCPStderrTests(unittest.TestCase):
    def test_noisy_stderr_does_not_deadlock(self):
        from lightlx.agent.mcp import MCPServer
        tmp = tempfile.TemporaryDirectory()
        script = Path(tmp.name) / "noisy_mcp.py"
        script.write_text(
            "import json, sys\n"
            "sys.stderr.write('warn\\n' * 50000)\n"
            "sys.stderr.flush()\n"
            "def read_msg():\n"
            "    headers = {}\n"
            "    while True:\n"
            "        line = sys.stdin.buffer.readline()\n"
            "        if not line:\n"
            "            return None\n"
            "        if line in (b'\\r\\n', b'\\n'):\n"
            "            break\n"
            "        if b':' in line:\n"
            "            k, v = line.decode().split(':', 1)\n"
            "            headers[k.strip().lower()] = v.strip()\n"
            "    n = int(headers.get('content-length') or 0)\n"
            "    return json.loads(sys.stdin.buffer.read(n))\n"
            "def write_msg(msg):\n"
            "    data = json.dumps(msg).encode()\n"
            "    sys.stdout.buffer.write(f'Content-Length: {len(data)}\\r\\n\\r\\n'.encode() + data)\n"
            "    sys.stdout.buffer.flush()\n"
            "while True:\n"
            "    msg = read_msg()\n"
            "    if msg is None:\n"
            "        break\n"
            "    mid = msg.get('id')\n"
            "    if mid is None:\n"
            "        continue\n"
            "    method = msg.get('method')\n"
            "    if method == 'initialize':\n"
            "        write_msg({'jsonrpc': '2.0', 'id': mid, 'result': {"
            "'protocolVersion': '2024-11-05', 'capabilities': {},"
            "'serverInfo': {'name': 'noisy'}}})\n"
            "    elif method == 'tools/list':\n"
            "        write_msg({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': []}})\n"
            "    else:\n"
            "        write_msg({'jsonrpc': '2.0', 'id': mid, 'result': {}})\n"
        )
        srv = MCPServer("noisy", {"command": sys.executable, "args": [str(script)]}, tmp.name)
        try:
            srv.start(timeout=8)
            self.assertEqual(srv.info.get("serverInfo", {}).get("name"), "noisy")
            self.assertEqual(srv.tools, [])
            self.assertIn("warn", srv._stderr_tail)
        finally:
            srv.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

