from pathlib import Path

from microcode.agent_loop import run_agent_turn
from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry, create_explore_tool_registry
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


def test_explore_registry_is_read_only() -> None:
    names = {tool.name for tool in create_explore_tool_registry().list()}
    assert names == {"list_files", "read_file", "grep_files"}
    assert "write_file" not in names
    assert "explore" not in names


def test_explore_returns_nested_summary(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "e1", "toolName": "list_files", "input": {}}],
            ),
            AgentStep(type="assistant", content="auth.py handles login."),
        ]
    )
    events: list[str] = []
    result = create_default_tool_registry().execute(
        "explore",
        {"task": "How does login work?"},
        ToolContext(
            cwd=str(tmp_path),
            model=model,
            on_tool_start=lambda name, _args: events.append(name),
        ),
    )
    assert result.ok
    assert result.output == "auth.py handles login."
    assert events == ["explore/list_files"]
    assert model.seen[0][0]["role"] == "system"
    assert "read-only" in str(model.seen[0][0]["content"]).lower() or "Explore" in str(
        model.seen[0][0]["content"]
    )


def test_parent_turn_only_keeps_the_summary(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "explore", "input": {"task": "What is here?"}}],
            ),
            AgentStep(
                type="tool_calls",
                calls=[{"id": "e1", "toolName": "list_files", "input": {}}],
            ),
            AgentStep(type="assistant", content="Workspace has notes.txt."),
            AgentStep(type="assistant", content="There is a notes file."),
        ]
    )
    messages = run_agent_turn(
        model=model,
        tools=create_default_tool_registry(),
        messages=[{"role": "user", "content": "what is in this repo?"}],
        cwd=str(tmp_path),
    )
    tool_results = [item for item in messages if item.get("role") == "tool_result"]
    assert tool_results[0]["toolName"] == "explore"
    assert tool_results[0]["content"] == "Workspace has notes.txt."
    assert messages[-1]["content"] == "There is a notes file."
    assert not any(item.get("toolName") == "list_files" for item in messages)
    parent_roles = [item.get("role") for item in model.seen[0]]
    assert "tool_result" not in parent_roles


def test_explore_without_model_errors(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "explore",
        {"task": "find auth"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "No model" in result.output


def test_mock_explore_keeps_inner_search_off_parent_history(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    messages = run_agent_turn(
        model=MockModelAdapter(),
        tools=create_default_tool_registry(),
        messages=[{"role": "user", "content": "/explore What files are here?"}],
        cwd=str(tmp_path),
    )
    tool_results = [item for item in messages if item.get("role") == "tool_result"]
    assert tool_results[0]["toolName"] == "explore"
    assert "notes.txt" in tool_results[0]["content"]
    assert not any(item.get("toolName") == "list_files" for item in messages)
    assert messages[-1]["content"].startswith("Explore summary:")
