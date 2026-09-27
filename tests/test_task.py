from pathlib import Path

from microcode.agent_loop import run_agent_turn
from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry, create_subagent_tool_registry
from microcode.tools.task import READ_ONLY_TOOLS
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


def test_subagent_registry_never_nests() -> None:
    names = {tool.name for tool in create_subagent_tool_registry(None).list()}
    assert "write_file" in names
    assert "task" not in names
    assert "explore" not in names


def test_plan_registry_is_read_only() -> None:
    names = {tool.name for tool in create_subagent_tool_registry(READ_ONLY_TOOLS).list()}
    assert names == {
        "list_files",
        "file_tree",
        "read_file",
        "grep_files",
        "find_symbols",
        "find_references",
    }


def test_task_plan_returns_nested_summary(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "p1", "toolName": "list_files", "input": {}}],
            ),
            AgentStep(type="assistant", content="Recommend reading auth.py first."),
        ]
    )
    events: list[str] = []
    result = create_default_tool_registry().execute(
        "task",
        {"description": "plan login", "prompt": "How should we change login?", "agent_type": "plan"},
        ToolContext(
            cwd=str(tmp_path),
            model=model,
            on_tool_start=lambda name, _args: events.append(name),
        ),
    )
    assert result.ok
    assert result.output == "Recommend reading auth.py first."
    assert events == ["plan/list_files"]
    assert "Plan" in str(model.seen[0][0]["content"])


def test_task_general_can_write_without_polluting_parent(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "1",
                        "toolName": "task",
                        "input": {
                            "description": "add note",
                            "prompt": "create notes.txt",
                            "agent_type": "general",
                        },
                    }
                ],
            ),
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "g1",
                        "toolName": "write_file",
                        "input": {"path": "notes.txt", "content": "hello\n"},
                    }
                ],
            ),
            AgentStep(type="assistant", content="Created notes.txt."),
            AgentStep(type="assistant", content="The note is in place."),
        ]
    )
    messages = run_agent_turn(
        model=model,
        tools=create_default_tool_registry(),
        messages=[{"role": "user", "content": "add a note"}],
        cwd=str(tmp_path),
    )
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
    tool_results = [item for item in messages if item.get("role") == "tool_result"]
    assert tool_results[0]["toolName"] == "task"
    assert tool_results[0]["content"] == "Created notes.txt."
    assert messages[-1]["content"] == "The note is in place."
    assert not any(item.get("toolName") == "write_file" for item in messages)


def test_task_rejects_unknown_type(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "task",
        {"description": "x", "agent_type": "reviewer"},
        ToolContext(cwd=str(tmp_path), model=ScriptedModel([])),
    )
    assert not result.ok
    assert "agent_type" in result.output


def test_mock_task_explore(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    messages = run_agent_turn(
        model=MockModelAdapter(),
        tools=create_default_tool_registry(),
        messages=[{"role": "user", "content": "/task explore What files are here?"}],
        cwd=str(tmp_path),
    )
    tool_results = [item for item in messages if item.get("role") == "tool_result"]
    assert tool_results[0]["toolName"] == "task"
    assert "notes.txt" in tool_results[0]["content"]
    assert not any(item.get("toolName") == "list_files" for item in messages)
