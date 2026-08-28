import json
from pathlib import Path

from microcode.permissions import PermissionStore, parse_decision, parse_permissions_command
from microcode.repl import run_repl, run_user_turn
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


def test_parse_decision() -> None:
    assert parse_decision(True) == (True, False)
    assert parse_decision(False) == (False, False)
    assert parse_decision("y") == (True, False)
    assert parse_decision("yes") == (True, False)
    assert parse_decision("a") == (True, True)
    assert parse_decision("always") == (True, True)
    assert parse_decision("n") == (False, False)
    assert parse_decision("") == (False, False)


def test_parse_permissions_command() -> None:
    assert parse_permissions_command("/permissions") == ("show", "", "")
    assert parse_permissions_command("/permissions clear") == ("clear", "", "")
    assert parse_permissions_command("/permissions remove write notes.txt") == (
        "remove",
        "write",
        "notes.txt",
    )
    assert parse_permissions_command("/permissions remove command pytest -q") == (
        "remove",
        "command",
        "pytest -q",
    )
    assert parse_permissions_command("/permissions remove") == ("help", "", "")
    assert parse_permissions_command("/permissions weird") == ("help", "", "")


def test_store_round_trip(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    assert store.allow("write", "notes.txt")
    assert store.allow("command", "pytest -q")
    assert not store.allow("write", "notes.txt")
    store.save()

    loaded = PermissionStore.load(tmp_path)
    assert loaded.is_allowed("write", "notes.txt")
    assert loaded.is_allowed("write", r"notes.txt")
    assert loaded.is_allowed("command", "pytest -q")
    assert not loaded.is_allowed("command", "pytest")
    data = json.loads((tmp_path / ".microcode" / "permissions.json").read_text(encoding="utf-8"))
    assert data == {"writes": ["notes.txt"], "commands": ["pytest -q"]}


def test_store_remove_and_clear(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    store.allow("write", "a.txt")
    store.allow("write", "b.txt")
    store.allow("command", "ls")
    assert store.remove("write", "a.txt")
    assert store.writes == ["b.txt"]
    store.clear()
    assert store.writes == []
    assert store.commands == []


def test_always_remembers_write_and_skips_next_prompt(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = PermissionStore.load(tmp_path)
    asked: list[str] = []

    def on_approve(summary: str) -> str:
        asked.append(summary)
        return "always"

    first = tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "one\n"},
        ToolContext(cwd=str(tmp_path), on_approve=on_approve, permissions=store),
    )
    second = tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "two\n"},
        ToolContext(
            cwd=str(tmp_path),
            on_approve=lambda _summary: asked.append("again") or False,
            permissions=store,
        ),
    )

    assert first.ok
    assert second.ok
    assert len(asked) == 1
    assert store.writes == ["hello.txt"]
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "two\n"
    assert json.loads(store.path.read_text(encoding="utf-8"))["writes"] == ["hello.txt"]


def test_yes_once_does_not_remember(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = PermissionStore.load(tmp_path)
    asked = 0

    def on_approve(_summary: str) -> bool:
        nonlocal asked
        asked += 1
        return True

    tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "one\n"},
        ToolContext(cwd=str(tmp_path), on_approve=on_approve, permissions=store),
    )
    tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "two\n"},
        ToolContext(cwd=str(tmp_path), on_approve=on_approve, permissions=store),
    )

    assert asked == 2
    assert store.writes == []
    assert not store.path.is_file()


def test_auto_approve_does_not_remember(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = PermissionStore.load(tmp_path)
    result = tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "hello\n"},
        ToolContext(cwd=str(tmp_path), permissions=store),
    )
    assert result.ok
    assert store.writes == []
    assert not store.path.is_file()


def test_remembered_command_skips_prompt(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    command = 'python -c "open(\'ran.txt\', \'w\').write(\'x\')"'
    store = PermissionStore.load(tmp_path)
    store.allow("command", command)
    asked: list[str] = []
    result = tools.execute(
        "run_command",
        {"command": command},
        ToolContext(
            cwd=str(tmp_path),
            on_approve=lambda summary: asked.append(summary) or False,
            permissions=store,
        ),
    )
    assert result.ok
    assert asked == []
    assert (tmp_path / "ran.txt").read_text(encoding="utf-8") == "x"


def test_always_remembers_command(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = PermissionStore.load(tmp_path)
    command = 'python -c "print(1)"'
    result = tools.execute(
        "run_command",
        {"command": command},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: "a", permissions=store),
    )
    assert result.ok
    assert store.commands == [command]
    assert json.loads(store.path.read_text(encoding="utf-8"))["commands"] == [command]


def test_user_turn_uses_remembered_write(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    store.allow("write", "notes.txt")
    asked: list[str] = []
    run_user_turn(
        model=ScriptedModel(
            [
                AgentStep(
                    type="tool_calls",
                    calls=[
                        {
                            "id": "1",
                            "toolName": "write_file",
                            "input": {"path": "notes.txt", "content": "hi\n"},
                        }
                    ],
                ),
                AgentStep(type="assistant", content="done"),
            ]
        ),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        user_text="write notes",
        cwd=str(tmp_path),
        on_approve=lambda summary: asked.append(summary) or False,
        permissions=store,
    )
    assert asked == []
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hi\n"


def test_repl_permissions_commands(tmp_path: Path, capsys) -> None:
    store = PermissionStore.load(tmp_path)
    store.allow("write", "notes.txt")
    store.allow("command", "pytest -q")
    store.save()
    lines = iter(
        [
            "/permissions",
            "/permissions remove write notes.txt",
            "/permissions remove write missing.txt",
            "/permissions weird",
            "/permissions clear",
            "/permissions",
            "/exit",
        ]
    )

    run_repl(
        model=ScriptedModel([]),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
        permissions=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    err = capsys.readouterr().err
    assert "notes.txt" in err
    assert "pytest -q" in err
    assert "Removed write notes.txt" in err
    assert "Not found: write missing.txt" in err
    assert "Usage:" in err
    assert "Cleared remembered permissions." in err
    assert "No remembered permissions." in err
    reloaded = PermissionStore.load(tmp_path)
    assert reloaded.writes == []
    assert reloaded.commands == []
