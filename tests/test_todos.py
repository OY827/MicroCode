from pathlib import Path

from microcode.repl import run_repl, run_user_turn
from microcode.todos import TodoItem, TodoStore
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def test_todo_write_replaces_the_whole_list(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = TodoStore()
    context = ToolContext(cwd=str(tmp_path), todos=store)
    first = tools.execute(
        "todo_write",
        {
            "todos": [
                {"content": "add function", "status": "in_progress"},
                {"content": "write tests"},
            ]
        },
        context,
    )
    second = tools.execute(
        "todo_write",
        {
            "todos": [
                {"content": "add function", "status": "completed"},
                {"content": "write tests", "status": "in_progress"},
                {"content": "run pytest", "status": "pending"},
            ]
        },
        context,
    )

    assert first.ok
    assert "[~] add function" in first.output
    assert "[ ] write tests" in first.output
    assert second.ok
    assert "[x] add function" in second.output
    assert "[~] write tests" in second.output
    assert "[ ] run pytest" in second.output
    assert "1 pending, 1 in progress, 1 completed" in second.output
    assert [item.content for item in store.items] == [
        "add function",
        "write tests",
        "run pytest",
    ]
    assert not any(tmp_path.rglob("todos*"))


def test_todo_write_rejects_bad_input(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    context = ToolContext(cwd=str(tmp_path), todos=TodoStore())
    result = tools.execute(
        "todo_write",
        {"todos": [{"content": "x", "status": "later"}]},
        context,
    )
    assert not result.ok
    assert "status" in result.output


def test_empty_list_clears_todos(tmp_path: Path) -> None:
    store = TodoStore(items=[TodoItem(content="old", status="pending")])
    result = create_default_tool_registry().execute(
        "todo_write",
        {"todos": []},
        ToolContext(cwd=str(tmp_path), todos=store),
    )
    assert result.ok
    assert result.output == "No todos."
    assert store.items == []


def test_user_turn_shares_the_store(tmp_path: Path) -> None:
    store = TodoStore()
    run_user_turn(
        model=ScriptedModel(
            [
                AgentStep(
                    type="tool_calls",
                    calls=[
                        {
                            "id": "1",
                            "toolName": "todo_write",
                            "input": {
                                "todos": [
                                    {"content": "add function", "status": "in_progress"}
                                ]
                            },
                        }
                    ],
                ),
                AgentStep(type="assistant", content="working"),
            ]
        ),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        user_text="plan it",
        cwd=str(tmp_path),
        todos=store,
    )
    assert store.items[0].content == "add function"
    assert store.items[0].status == "in_progress"


def test_repl_todos_command(tmp_path: Path, capsys) -> None:
    store = TodoStore()
    lines = iter(["plan it", "/todos", "/exit"])
    run_repl(
        model=ScriptedModel(
            [
                AgentStep(
                    type="tool_calls",
                    calls=[
                        {
                            "id": "1",
                            "toolName": "todo_write",
                            "input": {
                                "todos": [
                                    {"content": "add function", "status": "completed"},
                                    {"content": "run tests", "status": "pending"},
                                ]
                            },
                        }
                    ],
                ),
                AgentStep(type="assistant", content="working"),
            ]
        ),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
        todos=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    err = capsys.readouterr().err
    assert "[x] add function" in err
    assert "[ ] run tests" in err
    assert "1 pending, 0 in progress, 1 completed" in err
