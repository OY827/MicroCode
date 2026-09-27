from pathlib import Path

from microcode.agent_loop import INTERRUPTED_MESSAGE
from microcode.repl import handle_local_command, run_repl, run_user_turn
from microcode.session import SessionStore
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage, TokenUsage
from microcode.usage import UsageLedger


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.seen.append(list(messages))
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="echo",
                description="echo tool",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=lambda input_data, _context: ToolResult(
                    ok=True, output=f"echo:{input_data['text']}"
                ),
            )
        ]
    )


def test_handle_local_exit_and_help() -> None:
    assert handle_local_command("/exit") == "exit"
    assert handle_local_command("  /QUIT  ") == "exit"
    assert handle_local_command("/help") == "help"
    assert handle_local_command("/status") == "status"
    assert handle_local_command("/replay") == "replay"
    assert handle_local_command("/replay latest") == "replay"
    assert handle_local_command("/tape") == "tape"
    assert handle_local_command("/tape 5") == "tape"
    assert handle_local_command("/tape path") == "tape"
    assert handle_local_command("/fixture") == "fixture"
    assert handle_local_command("/model") == "model"
    assert handle_local_command("/model mock") == "model"
    assert handle_local_command("/cost") == "cost"
    assert handle_local_command("/session") == "session"
    assert handle_local_command("/sessions") == "sessions"
    assert handle_local_command("/undo") == "undo"
    assert handle_local_command("/redo") == "redo"
    assert handle_local_command("/mode") == "mode"
    assert handle_local_command("/mode read") == "mode"
    assert handle_local_command("/checkpoints") == "checkpoints"
    assert handle_local_command("/rewind") == "rewind"
    assert handle_local_command("/rewind cp_1") == "rewind"
    assert handle_local_command("/rewind-preview") == "rewind-preview"
    assert handle_local_command("/rewind-preview cp_1") == "rewind-preview"
    assert handle_local_command("/compact") == "compact"
    assert handle_local_command("/memory") == "memory"
    assert handle_local_command("/memory add Use pytest") == "memory"
    assert handle_local_command("/memory search pytest") == "memory"
    assert handle_local_command("/memory maintain") == "memory"
    assert handle_local_command("/skills") == "skills"
    assert handle_local_command("/mcp") == "mcp"
    assert handle_local_command("/permissions") == "permissions"
    assert handle_local_command("/permissions clear") == "permissions"
    assert handle_local_command("/permissions remove write notes.txt") == "permissions"
    assert handle_local_command("/todos") == "todos"
    assert handle_local_command("/jobs") == "jobs"
    assert handle_local_command("/readiness") == "readiness"
    assert handle_local_command("/init") == "init"
    assert handle_local_command("list files") is None


def test_run_user_turn_appends_and_keeps_history() -> None:
    model = ScriptedModel(
        [
            AgentStep(type="assistant", content="first"),
            AgentStep(type="assistant", content="second"),
        ]
    )
    messages = [{"role": "system", "content": "sys"}]
    messages = run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=messages,
        user_text="one",
        cwd=".",
    )
    messages = run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=messages,
        user_text="two",
        cwd=".",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[1]["content"] == "one"
    assert messages[3]["content"] == "two"
    assert messages[-1]["content"] == "second"
    assert any(item["role"] == "user" and item["content"] == "one" for item in model.seen[1])


def test_repl_stops_on_exit_and_runs_one_turn() -> None:
    model = ScriptedModel([AgentStep(type="assistant", content="hi")])
    lines = iter(["hello", "/exit"])

    messages = run_repl(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    assert model.calls == 1
    assert messages[-1] == {"role": "assistant", "content": "hi"}


class InterruptOnceModel:
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.seen.append(list(messages))
        self.calls += 1
        if self.calls == 1:
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "echo", "input": {"text": "hi"}}],
            )
        return AgentStep(type="assistant", content="continued")


def _interrupt_registry() -> ToolRegistry:
    def run_echo(_input_data: dict, _context) -> ToolResult:
        raise KeyboardInterrupt()

    return ToolRegistry(
        [
            ToolDefinition(
                name="echo",
                description="echo tool",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=run_echo,
            )
        ]
    )


def test_interrupt_keeps_history_for_next_turn() -> None:
    model = InterruptOnceModel()
    messages = [{"role": "system", "content": "sys"}]
    messages = run_user_turn(
        model=model,
        tools=_interrupt_registry(),
        messages=messages,
        user_text="do it",
        cwd=".",
    )
    assert messages[-1]["content"] == INTERRUPTED_MESSAGE

    messages = run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=messages,
        user_text="continue",
        cwd=".",
    )
    assert messages[-1]["content"] == "continued"
    assert any(
        item["role"] == "assistant" and item["content"] == INTERRUPTED_MESSAGE
        for item in model.seen[1]
    )


def test_interrupt_saves_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    run_user_turn(
        model=InterruptOnceModel(),
        tools=_interrupt_registry(),
        messages=session.messages,
        user_text="do it",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    loaded = store.load(session.id)
    assert loaded.messages[-1]["content"] == INTERRUPTED_MESSAGE
    assert loaded.messages[1]["content"] == "do it"


def test_repl_ctrl_c_at_prompt_stays_in_session() -> None:
    model = ScriptedModel([AgentStep(type="assistant", content="hi")])
    calls = {"n": 0}

    def read_line() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt()
        return "/exit\n"

    messages = run_repl(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
        read_line=read_line,
        on_assistant_message=lambda _text: None,
    )
    assert calls["n"] == 2
    assert model.calls == 0
    assert messages[-1]["role"] == "system"


def test_repl_interrupt_during_turn_then_continues() -> None:
    model = InterruptOnceModel()
    lines = iter(["do it", "continue", "/exit"])
    messages = run_repl(
        model=model,
        tools=_interrupt_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    assert model.calls == 2
    assert messages[-1]["content"] == "continued"
    assert any(
        item["role"] == "assistant" and item["content"] == INTERRUPTED_MESSAGE for item in messages
    )


def test_repl_replay_cost_and_model_switch(tmp_path: Path, capsys) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = ScriptedModel(
        [AgentStep(type="assistant", content="hi there", usage=TokenUsage(12, 4))]
    )
    lines = iter(["hello", "/cost", "/replay", "/model mock", "/model", "/exit"])

    run_repl(
        model=model,
        tools=_echo_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        read_line=lambda: next(lines) + "\n",
        session=session,
        store=store,
        on_assistant_message=lambda _text: None,
        usage=UsageLedger(),
    )

    err = capsys.readouterr().err
    assert "model calls: 1" in err
    assert "input tokens: 12" in err
    assert "[1] you" in err
    assert "hello" in err
    assert "hi there" in err
    assert "Switched to offline mock" in err
    assert "offline mock (not a live network model)" in err
    loaded = store.load(session.id)
    assert loaded.model_name == "mock"
    assert loaded.usage["calls"] == 1
