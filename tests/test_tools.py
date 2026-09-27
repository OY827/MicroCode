from pathlib import Path

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.workspace import resolve_tool_path


def test_list_files_sees_workspace(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "src").mkdir()
    tools = create_default_tool_registry()
    result = tools.execute("list_files", {}, ToolContext(cwd=str(tmp_path)))
    assert result.ok
    assert "file hello.txt" in result.output
    assert "dir  src" in result.output


def test_file_tree_shows_nested_structure(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "login.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute("file_tree", {}, ToolContext(cwd=str(tmp_path)))
    assert result.ok
    assert result.output.startswith(".")
    assert "file hello.txt" in result.output
    assert "dir  src" in result.output
    assert "dir  auth" in result.output
    assert "file login.py" in result.output
    assert "├── " in result.output or "└── " in result.output


def test_file_tree_respects_max_depth(tmp_path: Path) -> None:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "login.py").write_text("x\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "file_tree",
        {"max_depth": 1},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "dir  src" in result.output
    assert "login.py" not in result.output
    assert "..." in result.output


def test_file_tree_skips_env_and_git(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MICROCODE_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir()
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute("file_tree", {}, ToolContext(cwd=str(tmp_path)))
    assert result.ok
    assert ".env" not in result.output
    assert ".git" not in result.output
    assert "node_modules" not in result.output
    assert "hello.txt" in result.output


def test_file_tree_rejects_bad_depth(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "file_tree",
        {"max_depth": 0},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "max_depth" in result.output


def test_list_files_skips_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MICROCODE_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute("list_files", {}, ToolContext(cwd=str(tmp_path)))
    assert result.ok
    assert ".env" not in result.output
    assert "hello.txt" in result.output


def test_read_file_returns_contents(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello microcode", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "read_file",
        {"path": "notes.txt"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "hello microcode" in result.output


def test_path_cannot_escape_workspace(tmp_path: Path) -> None:
    context = ToolContext(cwd=str(tmp_path))
    try:
        resolve_tool_path(context, "../secret.txt")
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass


def test_grep_files_finds_line(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello microcode\nsecond line\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "grep_files",
        {"pattern": "microcode"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "notes.txt:1:hello microcode" in result.output


def test_write_file_creates_and_previews(tmp_path: Path) -> None:
    previews: list[tuple[str, str]] = []
    tools = create_default_tool_registry()
    result = tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "hello\n"},
        ToolContext(cwd=str(tmp_path), on_write_preview=lambda path, diff: previews.append((path, diff))),
    )
    assert result.ok
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    assert previews and previews[0][0] == "hello.txt"
    assert "+hello" in previews[0][1]


def test_edit_file_replaces_unique_text(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello world\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "edit_file",
        {"path": "notes.txt", "search": "world", "replace": "microcode"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello microcode\n"


def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("foo foo\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "edit_file",
        {"path": "notes.txt", "search": "foo", "replace": "bar"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "occurs 2 times" in result.output
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "foo foo\n"


def test_patch_file_applies_multiple_replacements(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello world\nbye moon\n", encoding="utf-8")
    previews: list[tuple[str, str]] = []
    tools = create_default_tool_registry()
    result = tools.execute(
        "patch_file",
        {
            "path": "notes.txt",
            "replacements": [
                {"search": "world", "replace": "microcode"},
                {"search": "moon", "replace": "sun"},
            ],
        },
        ToolContext(
            cwd=str(tmp_path),
            on_write_preview=lambda path, diff: previews.append((path, diff)),
        ),
    )
    assert result.ok
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello microcode\nbye sun\n"
    assert "2 replacement(s)" in result.output
    assert previews and "+hello microcode" in previews[0][1]


def test_patch_file_rolls_back_in_memory_when_a_hunk_fails(tmp_path: Path) -> None:
    original = "hello world\n"
    (tmp_path / "notes.txt").write_text(original, encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "patch_file",
        {
            "path": "notes.txt",
            "replacements": [
                {"search": "world", "replace": "microcode"},
                {"search": "missing", "replace": "x"},
            ],
        },
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "Replacement 2 not found" in result.output
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == original


def test_patch_file_rejects_ambiguous_hunk(tmp_path: Path) -> None:
    original = "foo foo\nbar\n"
    (tmp_path / "notes.txt").write_text(original, encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "patch_file",
        {
            "path": "notes.txt",
            "replacements": [
                {"search": "foo", "replace": "baz"},
                {"search": "bar", "replace": "qux"},
            ],
        },
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "Replacement 1 occurs 2 times" in result.output
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == original


def test_patch_file_replace_all_on_one_hunk(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("foo foo\nbar\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "patch_file",
        {
            "path": "notes.txt",
            "replacements": [
                {"search": "foo", "replace": "baz", "replace_all": True},
                {"search": "bar", "replace": "qux"},
            ],
        },
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "baz baz\nqux\n"


def test_patch_file_respects_rejection(tmp_path: Path) -> None:
    original = "hello world\n"
    (tmp_path / "notes.txt").write_text(original, encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "patch_file",
        {
            "path": "notes.txt",
            "replacements": [{"search": "world", "replace": "microcode"}],
        },
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == original


def test_run_command_captures_stdout(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "run_command",
        {"command": 'python -c "print(123)"'},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "123" in result.output
    assert "exit_code: 0" in result.output


def test_run_command_uses_workspace_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("workspace-ok", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "run_command",
        {"command": 'python -c "print(open(\'marker.txt\', encoding=\'utf-8\').read())"'},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "workspace-ok" in result.output


def test_run_command_nonzero_exit_is_error(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "run_command",
        {"command": 'python -c "raise SystemExit(2)"'},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "exit_code: 2" in result.output


def test_run_command_times_out(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "run_command",
        {"command": 'python -c "import time; time.sleep(5)"', "timeout": 1},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "timed out" in result.output


def test_write_file_can_be_rejected(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "write_file",
        {"path": "hello.txt", "content": "hello\n"},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output
    assert not (tmp_path / "hello.txt").exists()


def test_run_command_can_be_rejected(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "run_command",
        {"command": 'python -c "open(\'ran.txt\', \'w\').write(\'x\')"'},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output
    assert not (tmp_path / "ran.txt").exists()


def test_write_file_refuses_env(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    result = tools.execute(
        "write_file",
        {"path": ".env", "content": "MICROCODE_API_KEY=stolen\n"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "protected" in result.output
    assert not (tmp_path / ".env").exists()


def test_mock_model_lists_then_answers() -> None:
    model = MockModelAdapter()
    first = model.next([{"role": "user", "content": "/ls"}])
    assert first.type == "tool_calls"
    assert first.calls[0]["toolName"] == "list_files"

    second = model.next(
        [
            {"role": "user", "content": "/ls"},
            {
                "role": "assistant_tool_call",
                "toolName": "list_files",
                "toolUseId": "mock-1",
            },
            {
                "role": "tool_result",
                "toolName": "list_files",
                "content": "file hello.txt",
            },
        ]
    )
    assert second.type == "assistant"
    assert "hello.txt" in second.content


def test_mock_model_file_tree() -> None:
    model = MockModelAdapter()
    first = model.next([{"role": "user", "content": "/tree src::2"}])
    assert first.type == "tool_calls"
    assert first.calls[0]["toolName"] == "file_tree"
    assert first.calls[0]["input"] == {"path": "src", "max_depth": 2}


def test_mock_model_patch_file() -> None:
    model = MockModelAdapter()
    first = model.next(
        [{"role": "user", "content": "/patch notes.txt::world=>microcode::moon=>sun"}]
    )
    assert first.type == "tool_calls"
    assert first.calls[0]["toolName"] == "patch_file"
    assert first.calls[0]["input"] == {
        "path": "notes.txt",
        "replacements": [
            {"search": "world", "replace": "microcode"},
            {"search": "moon", "replace": "sun"},
        ],
    }
