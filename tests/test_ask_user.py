from pathlib import Path

from microcode.repl import run_user_turn
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.ask_user import NO_USER_MESSAGE
from microcode.types import AgentStep, ChatMessage


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


def test_ask_user_returns_answer(tmp_path: Path) -> None:
    asked: list[str] = []
    result = create_default_tool_registry().execute(
        "ask_user",
        {"question": "Python or text?"},
        ToolContext(
            cwd=str(tmp_path),
            on_ask_user=lambda question: asked.append(question) or "Python, greet(name)",
        ),
    )
    assert result.ok
    assert result.output == "Python, greet(name)"
    assert asked == ["Python or text?"]


def test_ask_user_without_callback_errors(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "ask_user",
        {"question": "Python or text?"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert result.output == NO_USER_MESSAGE


def test_ask_user_empty_answer_errors(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "ask_user",
        {"question": "Python or text?"},
        ToolContext(cwd=str(tmp_path), on_ask_user=lambda _question: "  "),
    )
    assert not result.ok
    assert "did not answer" in result.output


def test_ask_user_requires_question(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "ask_user",
        {},
        ToolContext(cwd=str(tmp_path), on_ask_user=lambda _question: "x"),
    )
    assert not result.ok
    assert "question" in result.output


def test_user_turn_feeds_answer_back_to_the_model(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "1",
                        "toolName": "ask_user",
                        "input": {"question": "Python or text?"},
                    }
                ],
            ),
            AgentStep(type="assistant", content="will write Python"),
        ]
    )
    messages = run_user_turn(
        model=model,
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        user_text="add greet",
        cwd=str(tmp_path),
        on_ask_user=lambda _question: "Python",
    )
    result = next(item for item in messages if item["role"] == "tool_result")
    assert result["content"] == "Python"
    assert result["isError"] is False
    assert messages[-1]["content"] == "will write Python"
    assert any(
        item["role"] == "tool_result" and item["content"] == "Python"
        for item in model.seen[1]
    )
