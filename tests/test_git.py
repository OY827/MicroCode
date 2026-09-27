import subprocess
from pathlib import Path

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry


def _git(cwd: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _init_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "microcode@example.com")
    _git(tmp_path, "config", "user.name", "MicroCode")


def test_git_status_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    result = create_default_tool_registry().execute(
        "git",
        {"action": "status"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "clean" in result.output.lower()


def test_git_status_lists_untracked(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "status"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "notes.txt" in result.output


def test_git_diff_and_commit(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.txt")
    _git(tmp_path, "commit", "-m", "first")
    (tmp_path / "notes.txt").write_text("hello world\n", encoding="utf-8")

    tools = create_default_tool_registry()
    context = ToolContext(cwd=str(tmp_path))
    diff = tools.execute("git", {"action": "diff"}, context)
    assert diff.ok
    assert "Unstaged:" in diff.output
    assert "+hello world" in diff.output or "+hello world\n" in diff.output + "\n"

    _git(tmp_path, "add", "notes.txt")
    committed = tools.execute(
        "git",
        {"action": "commit", "message": "second"},
        context,
    )
    assert committed.ok

    log = tools.execute("git", {"action": "log", "max_lines": 5}, context)
    assert log.ok
    assert "second" in log.output


def test_git_commit_requires_staged_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "commit", "message": "nope"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "No staged changes" in result.output


def test_git_commit_can_be_rejected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.txt")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "commit", "message": "secret"},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output
    log = create_default_tool_registry().execute(
        "git",
        {"action": "log"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not log.ok or "secret" not in log.output


def test_git_add_then_commit(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    tools = create_default_tool_registry()
    context = ToolContext(cwd=str(tmp_path))
    added = tools.execute("git", {"action": "add", "paths": ["notes.txt"]}, context)
    assert added.ok
    assert "notes.txt" in added.output
    committed = tools.execute("git", {"action": "commit", "message": "first"}, context)
    assert committed.ok
    log = tools.execute("git", {"action": "log"}, context)
    assert log.ok
    assert "first" in log.output


def test_git_add_accepts_single_path(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add", "path": "notes.txt"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    status = create_default_tool_registry().execute(
        "git",
        {"action": "status"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert "notes.txt" in status.output


def test_git_add_requires_paths(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "paths" in result.output


def test_git_add_rejects_flags(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add", "paths": ["-A"]},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "flags" in result.output


def test_git_add_rejects_env_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add", "paths": [".env"]},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "protected" in result.output


def test_git_add_rejects_folder_containing_env(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / ".env").write_text("SECRET=1\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add", "paths": ["pkg"]},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "protected" in result.output


def test_git_add_can_be_rejected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "git",
        {"action": "add", "paths": ["notes.txt"]},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output
    status = create_default_tool_registry().execute(
        "git",
        {"action": "status"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert "??" in status.output or "notes.txt" in status.output


def test_mock_model_git_add() -> None:
    step = MockModelAdapter().next([{"role": "user", "content": "/git add notes.txt app.py"}])
    assert step.type == "tool_calls"
    assert step.calls[0]["input"] == {"action": "add", "paths": ["notes.txt", "app.py"]}


def test_git_rejects_unknown_action(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "git",
        {"action": "push"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "action" in result.output


def test_mock_model_git_status() -> None:
    model = MockModelAdapter()
    step = model.next([{"role": "user", "content": "/git status"}])
    assert step.type == "tool_calls"
    assert step.calls[0]["toolName"] == "git"
    assert step.calls[0]["input"] == {"action": "status"}
