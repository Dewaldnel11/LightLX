import json
import os
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

    def test_sanitize_drops_orphan_tools(self):
        from lightlx.agent.context import sanitize_messages
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
        # empty completions with no tools end as done (nudged or not)
        self.assertIn(result.status, ("done", "max_iters"))


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


if __name__ == "__main__":
    unittest.main()
