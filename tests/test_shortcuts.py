import subprocess
from pathlib import Path

from microcode.repl import handle_local_command, run_repl
from microcode.session import SessionStore
from microcode.shortcuts import parse_tool_shortcut
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.types import AgentStep, ChatMessage


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.calls += 1
        return AgentStep(type="assistant", content="model ran")


def test_parse_ls_and_tree() -> None:
    assert parse_tool_shortcut("/ls") is not None
    parsed = parse_tool_shortcut("/ls")
    assert getattr(parsed, "tool_name", None) == "list_files"
    assert parse_tool_shortcut("/ls src").input_data == {"path": "src"}
    assert parse_tool_shortcut("/lsfoo") is None
    tree = parse_tool_shortcut("/tree src::2")
    assert tree.tool_name == "file_tree"
    assert tree.input_data == {"path": "src", "max_depth": 2}


def test_parse_grep_and_read() -> None:
    grep = parse_tool_shortcut("/grep microcode::src")
    assert grep.tool_name == "grep_files"
    assert grep.input_data == {"pattern": "microcode", "path": "src"}
    assert parse_tool_shortcut("/grep") == "Usage: /grep <pattern>[::path]"
    assert parse_tool_shortcut("/read notes.txt").input_data == {"path": "notes.txt"}
    assert parse_tool_shortcut("/read") == "Usage: /read <path>"


def test_parse_write_edit_patch() -> None:
    write = parse_tool_shortcut("/write notes.txt::hello")
    assert write.tool_name == "write_file"
    assert write.input_data == {"path": "notes.txt", "content": "hello"}
    assert parse_tool_shortcut("/write") == "Usage: /write <path>::<content>"
    assert parse_tool_shortcut("/writefoo") is None
    edit = parse_tool_shortcut("/edit app.py::timeout = 30::timeout = 60")
    assert edit.tool_name == "edit_file"
    assert edit.input_data == {
        "path": "app.py",
        "search": "timeout = 30",
        "replace": "timeout = 60",
    }
    assert parse_tool_shortcut("/edit app.py::only-two") == "Usage: /edit <path>::<search>::<replace>"
    patch = parse_tool_shortcut("/patch notes.txt::world=>microcode::moon=>sun")
    assert patch.tool_name == "patch_file"
    assert patch.input_data == {
        "path": "notes.txt",
        "replacements": [
            {"search": "world", "replace": "microcode"},
            {"search": "moon", "replace": "sun"},
        ],
    }


def test_parse_copy_move_delete_cmd() -> None:
    copy = parse_tool_shortcut("/copy a.txt::b.txt")
    assert copy.tool_name == "batch_copy"
    assert copy.input_data == {"source": "a.txt", "destination": "b.txt"}
    move = parse_tool_shortcut("/move a.txt::b.txt")
    assert move.tool_name == "batch_move"
    delete = parse_tool_shortcut("/delete pkg::recursive")
    assert delete.tool_name == "batch_delete"
    assert delete.input_data == {"path": "pkg", "recursive": True}
    cmd = parse_tool_shortcut("/cmd pytest -q")
    assert cmd.tool_name == "run_command"
    assert cmd.input_data == {"command": "pytest -q"}
    nested = parse_tool_shortcut("/cmd src::pytest -q")
    assert nested.input_data == {"command": "pytest -q", "cwd": "src"}
    git_add = parse_tool_shortcut("/git add notes.txt app.py")
    assert git_add.tool_name == "git"
    assert git_add.input_data == {"action": "add", "paths": ["notes.txt", "app.py"]}
    assert parse_tool_shortcut("/git add") == "Usage: /git add <path> [path...]"
    commit = parse_tool_shortcut("/git commit first note")
    assert commit.input_data == {"action": "commit", "message": "first note"}


def test_shortcuts_are_not_session_commands() -> None:
    assert handle_local_command("/ls") is None
    assert handle_local_command("/tree") is None
    assert handle_local_command("/grep foo") is None
    assert handle_local_command("/read notes.txt") is None
    assert handle_local_command("/write notes.txt::hi") is None
    assert handle_local_command("/edit a.py::old::new") is None
    assert handle_local_command("/patch a.py::old=>new") is None
    assert handle_local_command("/copy a.txt::b.txt") is None
    assert handle_local_command("/move a.txt::b.txt") is None
    assert handle_local_command("/delete a.txt") is None
    assert handle_local_command("/cmd echo hi") is None
    assert handle_local_command("/git add notes.txt") is None


def test_repl_ls_does_not_call_the_model(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = CountingModel()
    printed: list[str] = []
    lines = iter(["/ls", "/exit"])

    run_repl(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=printed.append,
    )

    assert model.calls == 0
    assert printed == []
    loaded = store.load(session.id)
    assert loaded.messages == [{"role": "system", "content": "sys"}]


def test_repl_read_and_grep(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello microcode\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    model = CountingModel()
    lines = iter(["/read notes.txt", "/grep microcode", "/tree", "/exit"])

    run_repl(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    assert model.calls == 0


def test_repl_write_and_edit_do_not_call_the_model(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("timeout = 30\nprint('ok')\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = CountingModel()
    lines = iter(
        [
            "/write notes.txt::hello",
            "/edit app.py::timeout = 30::timeout = 60",
            "/exit",
        ]
    )

    run_repl(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    assert model.calls == 0
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "timeout = 60\nprint('ok')\n"
    loaded = store.load(session.id)
    assert loaded.messages == [{"role": "system", "content": "sys"}]
    assert loaded.checkpoint


def test_repl_git_add_does_not_call_the_model(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "microcode@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "MicroCode"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = CountingModel()
    lines = iter(["/git add notes.txt", "/exit"])

    run_repl(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    assert model.calls == 0
    status = create_default_tool_registry().execute(
        "git",
        {"action": "status"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert "notes.txt" in status.output
    assert "A  notes.txt" in status.output or "A notes.txt" in status.output
