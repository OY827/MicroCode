from pathlib import Path

from microcode.checkpoint import WriteCheckpoint
from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry


def _execute(tmp_path: Path, name: str, payload: dict, checkpoint: WriteCheckpoint | None = None, **kwargs):
    return create_default_tool_registry().execute(
        name,
        payload,
        ToolContext(cwd=str(tmp_path), checkpoint=checkpoint, **kwargs),
    )


def test_copy_file_and_undo(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(
        tmp_path,
        "batch_copy",
        {"source": "a.txt", "destination": "b.txt"},
        checkpoint,
    )
    assert result.ok
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "hello\n"
    report = "\n".join(checkpoint.restore(str(tmp_path)))
    assert "deleted b.txt" in report
    assert not (tmp_path / "b.txt").exists()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_copy_overwrites_file_and_undo(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("old\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(
        tmp_path,
        "batch_copy",
        {"source": "a.txt", "destination": "b.txt"},
        checkpoint,
    )
    assert result.ok
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new\n"
    checkpoint.restore(str(tmp_path))
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "old\n"


def test_copy_directory_and_undo(tmp_path: Path) -> None:
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "mod.py").write_text("x = 1\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(
        tmp_path,
        "batch_copy",
        {"source": "pkg", "destination": "pkg2"},
        checkpoint,
    )
    assert result.ok
    assert (tmp_path / "pkg2" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    checkpoint.restore(str(tmp_path))
    assert not (tmp_path / "pkg2").exists()
    assert (src / "mod.py").exists()


def test_copy_refuses_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg2").mkdir()
    result = _execute(tmp_path, "batch_copy", {"source": "pkg", "destination": "pkg2"})
    assert not result.ok
    assert "already exists" in result.output


def test_copy_refuses_destination_inside_source(tmp_path: Path) -> None:
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "mod.py").write_text("x\n", encoding="utf-8")
    result = _execute(tmp_path, "batch_copy", {"source": "pkg", "destination": "pkg/nested"})
    assert not result.ok
    assert "inside the source" in result.output


def test_move_file_and_undo(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(
        tmp_path,
        "batch_move",
        {"source": "a.txt", "destination": "sub/b.txt"},
        checkpoint,
    )
    assert result.ok
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "sub" / "b.txt").read_text(encoding="utf-8") == "hello\n"
    checkpoint.restore(str(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert not (tmp_path / "sub" / "b.txt").exists()


def test_move_refuses_existing_destination(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    result = _execute(tmp_path, "batch_move", {"source": "a.txt", "destination": "b.txt"})
    assert not result.ok
    assert "already exists" in result.output


def test_delete_file_and_undo(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(tmp_path, "batch_delete", {"path": "a.txt"}, checkpoint)
    assert result.ok
    assert not (tmp_path / "a.txt").exists()
    checkpoint.restore(str(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_delete_directory_requires_recursive(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x\n", encoding="utf-8")
    result = _execute(tmp_path, "batch_delete", {"path": "pkg"})
    assert not result.ok
    assert "recursive" in result.output
    assert (tmp_path / "pkg" / "mod.py").exists()


def test_delete_directory_and_undo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    result = _execute(
        tmp_path,
        "batch_delete",
        {"path": "pkg", "recursive": True},
        checkpoint,
    )
    assert result.ok
    assert not pkg.exists()
    checkpoint.restore(str(tmp_path))
    assert (pkg / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_refuses_protected_env(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    result = _execute(tmp_path, "batch_delete", {"path": ".env"})
    assert not result.ok
    assert "protected" in result.output
    assert (tmp_path / ".env").exists()


def test_refuses_path_escape(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "batch_copy",
        {"source": "a.txt", "destination": "../outside.txt"},
    )
    assert not result.ok
    assert "escapes workspace" in result.output.lower() or "error" in result.output.lower()


def test_user_can_reject(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    result = _execute(
        tmp_path,
        "batch_delete",
        {"path": "a.txt"},
        on_approve=lambda _: False,
    )
    assert not result.ok
    assert "rejected" in result.output.lower()
    assert (tmp_path / "a.txt").exists()


def test_mock_copy_move_delete() -> None:
    copy = MockModelAdapter().next([{"role": "user", "content": "/copy a.txt::b.txt"}])
    assert copy.calls[0]["toolName"] == "batch_copy"
    assert copy.calls[0]["input"] == {"source": "a.txt", "destination": "b.txt"}

    move = MockModelAdapter().next([{"role": "user", "content": "/move a.txt::b.txt"}])
    assert move.calls[0]["toolName"] == "batch_move"

    delete = MockModelAdapter().next([{"role": "user", "content": "/delete pkg::recursive"}])
    assert delete.calls[0]["toolName"] == "batch_delete"
    assert delete.calls[0]["input"] == {"path": "pkg", "recursive": True}
