import json

from microcode.agent_loop import run_agent_turn
from microcode.repl import apply_local_command
from microcode.shortcuts import ToolShortcut, execute_tool_shortcut
from microcode.tooling import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from microcode.turn_tape import TurnTape, format_tape, load_tape, tape_path
from microcode.types import AgentStep


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages, on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


class BoomModel:
    def next(self, messages, on_text_delta=None) -> AgentStep:
        raise RuntimeError("secret-token-should-not-be-logged")


def _echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=lambda data, _context: ToolResult(ok=True, output=f"echo:{data['text']}"),
            )
        ]
    )


def test_agent_turn_appends_model_and_tool_lines(tmp_path) -> None:
    tape = TurnTape(tmp_path, session_id="sess-1")
    run_agent_turn(
        model=ScriptedModel(
            [
                AgentStep(
                    type="tool_calls",
                    calls=[{"id": "1", "toolName": "echo", "input": {"text": "hi"}}],
                ),
                AgentStep(type="assistant", content="done"),
            ]
        ),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "say hi"}],
        cwd=str(tmp_path),
        tape=tape,
    )

    rows = load_tape(tmp_path)
    assert [row["kind"] for row in rows] == ["model", "tool", "model"]
    assert rows[0]["decision"] == "tool_calls"
    assert rows[0]["calls"] == ["echo text=hi"]
    assert rows[0]["session"] == "sess-1"
    assert rows[1]["tool"] == "echo"
    assert rows[1]["ok"] is True
    assert rows[1]["output"] == "echo:hi"
    assert rows[2]["content"] == "done"
    rendered = format_tape(tmp_path)
    assert "showing 1-3 of 3" in rendered
    assert "tool echo [ok]" in rendered
    assert "-> done" in rendered


def test_api_error_logs_type_only(tmp_path) -> None:
    tape = TurnTape(tmp_path)
    run_agent_turn(
        model=BoomModel(),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "go"}],
        cwd=str(tmp_path),
        tape=tape,
    )
    raw = tape_path(tmp_path).read_text(encoding="utf-8")
    assert "secret-token-should-not-be-logged" not in raw
    rows = load_tape(tmp_path)
    assert rows[0]["kind"] == "stop"
    assert rows[0]["reason"] == "api_error"
    assert rows[0]["detail"] == "RuntimeError"


def test_tape_command_shows_recent_lines(tmp_path) -> None:
    tape = TurnTape(tmp_path, session_id="s")
    tape.record_user("first")
    tape.record_user("second")
    tape.record_user("third")
    from microcode.mock_model import MockModelAdapter

    shown = apply_local_command(
        "/tape 2",
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
    )
    assert shown.handled is True
    assert "showing 2-3 of 3" in shown.output
    assert "first" not in shown.output
    assert "third" in shown.output

    path_only = apply_local_command(
        "/tape path",
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=[],
        cwd=str(tmp_path),
    )
    assert path_only.output == str(tape_path(tmp_path))


def test_shortcut_appends_tool_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_TURN_TAPE", "1")
    result = execute_tool_shortcut(
        ToolShortcut("echo", {"text": "yo"}),
        tools=_echo_registry(),
        cwd=str(tmp_path),
    )
    assert result.ok is True
    rows = load_tape(tmp_path)
    assert rows[0]["kind"] == "tool"
    assert rows[0]["via"] == "shortcut"
    assert rows[0]["output"] == "echo:yo"


def test_disabled_env_skips_user_turn_tape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_TURN_TAPE", "0")
    from microcode.repl import run_user_turn

    run_user_turn(
        model=ScriptedModel([AgentStep(type="assistant", content="hi")]),
        tools=_echo_registry(),
        messages=[{"role": "system", "content": "sys"}],
        user_text="hello",
        cwd=str(tmp_path),
    )
    assert load_tape(tmp_path) == []


def test_enabled_user_turn_writes_user_then_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_TURN_TAPE", "1")
    from microcode.repl import run_user_turn

    run_user_turn(
        model=ScriptedModel([AgentStep(type="assistant", content="hi")]),
        tools=_echo_registry(),
        messages=[{"role": "system", "content": "sys"}],
        user_text="hello\nthere",
        cwd=str(tmp_path),
    )
    rows = load_tape(tmp_path)
    assert [row["kind"] for row in rows] == ["user", "model"]
    assert rows[0]["text"] == "hello there"
    line = tape_path(tmp_path).read_text(encoding="utf-8").splitlines()[0]
    json.loads(line)


def test_subagent_marks_via(tmp_path) -> None:
    parent = TurnTape(tmp_path, session_id="parent")

    def run_task(data, context: ToolContext) -> ToolResult:
        from microcode.tools.task import run_subagent

        return run_subagent("explore", "look around", context)

    registry = ToolRegistry(
        [
            ToolDefinition(
                name="task",
                description="task",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=run_task,
            ),
            ToolDefinition(
                name="list_files",
                description="list",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=lambda _data, _context: ToolResult(ok=True, output="notes.txt"),
            ),
        ]
    )

    class ExploreThenAnswer:
        def __init__(self) -> None:
            self.calls = 0

        def next(self, messages, on_text_delta=None) -> AgentStep:
            self.calls += 1
            if self.calls == 1:
                return AgentStep(
                    type="tool_calls",
                    calls=[{"id": "t", "toolName": "task", "input": {"prompt": "look around"}}],
                )
            # Sub-agent shares this adapter. First nested call lists files, second summarizes.
            if any(item.get("role") == "tool_result" for item in messages):
                return AgentStep(type="assistant", content="found notes")
            if messages and messages[0].get("role") == "system" and "Explore" in str(messages[0].get("content")):
                return AgentStep(
                    type="tool_calls",
                    calls=[{"id": "n", "toolName": "list_files", "input": {}}],
                )
            return AgentStep(type="assistant", content="parent done")

    run_agent_turn(
        model=ExploreThenAnswer(),
        tools=registry,
        messages=[{"role": "user", "content": "explore"}],
        cwd=str(tmp_path),
        tape=parent,
    )
    rows = load_tape(tmp_path)
    nested = [row for row in rows if row.get("via") == "subagent"]
    assert nested
    assert nested[0]["kind"] == "user"
    assert any(row.get("kind") == "tool" and row.get("tool") == "list_files" for row in nested)
