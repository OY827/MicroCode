from microcode.repl import handle_local_command, run_repl, run_user_turn
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    def next(self, messages: list[ChatMessage]) -> AgentStep:
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
