from microcode.agent_loop import INTERRUPTED_MESSAGE, run_agent_turn
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _echo_registry() -> ToolRegistry:
    def run_echo(input_data: dict, _context) -> ToolResult:
        return ToolResult(ok=True, output=f"echo:{input_data['text']}")

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


def test_agent_turn_executes_tool_then_answers() -> None:
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "echo", "input": {"text": "hi"}}],
            ),
            AgentStep(type="assistant", content="done"),
        ]
    )

    messages = run_agent_turn(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "say hi"}],
        cwd=".",
    )

    assert messages[-1] == {"role": "assistant", "content": "done"}
    assert any(message["role"] == "tool_result" for message in messages)
    tool_result = next(message for message in messages if message["role"] == "tool_result")
    assert tool_result["content"] == "echo:hi"


def test_agent_turn_emits_callbacks() -> None:
    events: list[tuple[str, str]] = []
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "echo", "input": {"text": "hi"}}],
            ),
            AgentStep(type="assistant", content="done"),
        ]
    )

    run_agent_turn(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "say hi"}],
        cwd=".",
        on_tool_start=lambda name, _args: events.append(("start", name)),
        on_tool_result=lambda name, _output, _err: events.append(("result", name)),
        on_assistant_message=lambda text: events.append(("assistant", text)),
    )

    assert events == [
        ("start", "echo"),
        ("result", "echo"),
        ("assistant", "done"),
    ]


def test_unknown_tool_is_returned_as_error_result() -> None:
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "missing", "input": {}}],
            ),
            AgentStep(type="assistant", content="could not find it"),
        ]
    )

    messages = run_agent_turn(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "do a thing"}],
        cwd=".",
    )

    result = next(message for message in messages if message["role"] == "tool_result")
    assert result["isError"] is True
    assert "Unknown tool" in result["content"]


class ExplodingModel:
    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        raise RuntimeError("401 unauthorized")


class StreamingModel:
    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        if on_text_delta:
            on_text_delta("Hel")
            on_text_delta("lo")
        return AgentStep(type="assistant", content="Hello")


def test_streamed_text_does_not_reprint_assistant() -> None:
    printed: list[str] = []
    deltas: list[str] = []
    run_agent_turn(
        model=StreamingModel(),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "hi"}],
        cwd=".",
        on_assistant_message=printed.append,
        on_text_delta=deltas.append,
    )
    assert printed == []
    assert deltas == ["Hel", "lo", "\n"]


def test_agent_turn_surfaces_model_errors() -> None:
    messages = run_agent_turn(
        model=ExplodingModel(),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "hello"}],
        cwd=".",
    )
    assert messages[-1]["role"] == "assistant"
    assert "401 unauthorized" in messages[-1]["content"]


class InterruptModel:
    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        raise KeyboardInterrupt()


class InterruptDuringStream:
    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        if on_text_delta:
            on_text_delta("partial")
        raise KeyboardInterrupt()


def _interrupt_registry() -> ToolRegistry:
    def run_echo(input_data: dict, _context) -> ToolResult:
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


def test_interrupt_during_model_returns_without_raising() -> None:
    printed: list[str] = []
    messages = run_agent_turn(
        model=InterruptModel(),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "hello"}],
        cwd=".",
        on_assistant_message=printed.append,
    )
    assert messages[-1] == {"role": "assistant", "content": INTERRUPTED_MESSAGE}
    assert printed == [INTERRUPTED_MESSAGE]


def test_interrupt_during_stream_ends_the_line() -> None:
    deltas: list[str] = []
    messages = run_agent_turn(
        model=InterruptDuringStream(),
        tools=_echo_registry(),
        messages=[{"role": "user", "content": "hello"}],
        cwd=".",
        on_text_delta=deltas.append,
        on_assistant_message=lambda _text: None,
    )
    assert deltas == ["partial", "\n"]
    assert messages[-1]["content"] == INTERRUPTED_MESSAGE


def test_interrupt_during_tool_records_error_result() -> None:
    events: list[tuple[str, str]] = []
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[
                    {"id": "1", "toolName": "echo", "input": {"text": "hi"}},
                    {"id": "2", "toolName": "echo", "input": {"text": "skip"}},
                ],
            ),
            AgentStep(type="assistant", content="should not run"),
        ]
    )
    messages = run_agent_turn(
        model=model,
        tools=_interrupt_registry(),
        messages=[{"role": "user", "content": "say hi"}],
        cwd=".",
        on_tool_start=lambda name, _args: events.append(("start", name)),
        on_tool_result=lambda name, output, err: events.append(("result", output)),
        on_assistant_message=lambda text: events.append(("assistant", text)),
    )
    assert model.calls == 1
    assert events == [
        ("start", "echo"),
        ("result", INTERRUPTED_MESSAGE),
        ("assistant", INTERRUPTED_MESSAGE),
    ]
    result = next(message for message in messages if message["role"] == "tool_result")
    assert result["isError"] is True
    assert result["content"] == INTERRUPTED_MESSAGE
    assert messages[-1]["content"] == INTERRUPTED_MESSAGE
    assert sum(1 for message in messages if message["role"] == "tool_result") == 1
