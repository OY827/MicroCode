from __future__ import annotations

import os
import shutil
from pathlib import Path

from microcode.file_edit import PROTECTED_NAMES
from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import relative_workspace_path, resolve_tool_path, workspace_root

MAX_SNAPSHOT_FILES = 400


def _validate_pair(input_data: dict) -> dict:
    source = input_data.get("source")
    destination = input_data.get("destination")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source is required")
    if not isinstance(destination, str) or not destination.strip():
        raise ValueError("destination is required")
    return {"source": source.strip(), "destination": destination.strip()}


def _validate_delete(input_data: dict) -> dict:
    path = input_data.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    recursive = input_data.get("recursive", False)
    if not isinstance(recursive, bool):
        raise ValueError("recursive must be a boolean")
    return {"path": path.strip(), "recursive": recursive}


def _rel(cwd: str, target: Path) -> str:
    return relative_workspace_path(cwd, target)


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_protected(path: Path) -> bool:
    return path.name in PROTECTED_NAMES


def _find_protected(root: Path) -> Path | None:
    if _is_protected(root):
        return root
    if root.is_file():
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + filenames:
            if name in PROTECTED_NAMES:
                return Path(dirpath) / name
    return None


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            found.append(Path(dirpath) / name)
            if len(found) > MAX_SNAPSHOT_FILES:
                return found
    return found


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _snapshot_file(context, path: Path) -> str | None:
    if context.checkpoint is None:
        return None
    content = _read_text(path)
    if content is None:
        return f"Refusing to mutate binary file: {_rel(context.cwd, path)}"
    context.checkpoint.record_target(context.cwd, path, existed=True, content=content)
    return None


def _snapshot_tree(context, root: Path) -> str | None:
    files = _iter_files(root)
    if len(files) > MAX_SNAPSHOT_FILES:
        return f"Too many files to snapshot ({MAX_SNAPSHOT_FILES} max): {_rel(context.cwd, root)}"
    for file_path in files:
        error = _snapshot_file(context, file_path)
        if error:
            return error
    if context.checkpoint is not None and root.is_dir():
        context.checkpoint.record_target(context.cwd, root, existed=True, is_dir=True)
    return None


def _record_created(context, path: Path, *, is_dir: bool) -> None:
    if context.checkpoint is None:
        return
    context.checkpoint.record_target(context.cwd, path, existed=False, is_dir=is_dir)


def _approve(context, title: str, preview: str, key: str) -> ToolResult | None:
    if context.on_write_preview:
        context.on_write_preview(key, preview)
    if not context.approve(f"{title}\n{preview}", kind="write", key=key):
        return ToolResult(ok=False, output=context.reject_text(title.lower()))
    return None


def _run_copy(input_data: dict, context) -> ToolResult:
    src = resolve_tool_path(context, input_data["source"])
    dst = resolve_tool_path(context, input_data["destination"])
    if not src.exists():
        return ToolResult(ok=False, output=f"Source does not exist: {input_data['source']}")
    if src == dst:
        return ToolResult(ok=False, output="Source and destination are the same path")
    if src == workspace_root(context.cwd):
        return ToolResult(ok=False, output="Refusing to copy the workspace root")
    protected = _find_protected(src)
    if protected is not None or _is_protected(dst):
        return ToolResult(ok=False, output="Refusing to copy a protected file")
    if src.is_dir() and _is_within(src, dst):
        return ToolResult(ok=False, output="Destination is inside the source directory")

    src_rel = _rel(context.cwd, src)
    dst_rel = _rel(context.cwd, dst)
    if src.is_file() and dst.exists() and dst.is_dir():
        return ToolResult(ok=False, output=f"Destination is a directory: {input_data['destination']}")
    if src.is_dir() and dst.exists():
        return ToolResult(ok=False, output=f"Destination already exists: {input_data['destination']}")

    kind = "directory" if src.is_dir() else "file"
    preview = f"copy {kind} {src_rel} -> {dst_rel}"
    rejected = _approve(context, f"Copy {src_rel}?", preview, dst_rel)
    if rejected:
        return rejected

    if src.is_file():
        if dst.exists():
            error = _snapshot_file(context, dst)
            if error:
                return ToolResult(ok=False, output=error)
        else:
            _record_created(context, dst, is_dir=False)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError as error:
            return ToolResult(ok=False, output=str(error))
        return ToolResult(ok=True, output=f"Copied file {src_rel} -> {dst_rel}")

    _record_created(context, dst, is_dir=True)
    try:
        shutil.copytree(src, dst)
    except OSError as error:
        return ToolResult(ok=False, output=str(error))
    return ToolResult(ok=True, output=f"Copied directory {src_rel} -> {dst_rel}")


def _run_move(input_data: dict, context) -> ToolResult:
    src = resolve_tool_path(context, input_data["source"])
    dst = resolve_tool_path(context, input_data["destination"])
    if not src.exists():
        return ToolResult(ok=False, output=f"Source does not exist: {input_data['source']}")
    if src == dst:
        return ToolResult(ok=False, output="Source and destination are the same path")
    if src == workspace_root(context.cwd):
        return ToolResult(ok=False, output="Refusing to move the workspace root")
    protected = _find_protected(src)
    if protected is not None or _is_protected(dst):
        return ToolResult(ok=False, output="Refusing to move a protected file")
    if src.is_dir() and _is_within(src, dst):
        return ToolResult(ok=False, output="Destination is inside the source directory")
    if dst.exists():
        return ToolResult(ok=False, output=f"Destination already exists: {input_data['destination']}")

    src_rel = _rel(context.cwd, src)
    dst_rel = _rel(context.cwd, dst)
    kind = "directory" if src.is_dir() else "file"
    preview = f"move {kind} {src_rel} -> {dst_rel}"
    rejected = _approve(context, f"Move {src_rel}?", preview, dst_rel)
    if rejected:
        return rejected

    error = _snapshot_tree(context, src)
    if error:
        return ToolResult(ok=False, output=error)
    _record_created(context, dst, is_dir=src.is_dir())
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as error:
        return ToolResult(ok=False, output=str(error))
    return ToolResult(ok=True, output=f"Moved {kind} {src_rel} -> {dst_rel}")


def _run_delete(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    if target == workspace_root(context.cwd):
        return ToolResult(ok=False, output="Refusing to delete the workspace root")
    if _find_protected(target) is not None:
        return ToolResult(ok=False, output="Refusing to delete a protected file")
    if target.is_dir() and not input_data["recursive"]:
        return ToolResult(ok=False, output="Use recursive=true to delete directories")

    rel = _rel(context.cwd, target)
    kind = "directory" if target.is_dir() else "file"
    preview = f"delete {kind} {rel}"
    rejected = _approve(context, f"Delete {rel}?", preview, rel)
    if rejected:
        return rejected

    error = _snapshot_tree(context, target)
    if error:
        return ToolResult(ok=False, output=error)
    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as error:
        return ToolResult(ok=False, output=str(error))
    return ToolResult(ok=True, output=f"Deleted {kind} {rel}")


batch_copy_tool = ToolDefinition(
    name="batch_copy",
    description=(
        "Copy a file or directory inside the workspace. Destination directories must not already exist. "
        "Do not use run_command for cp."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    },
    validator=_validate_pair,
    run=_run_copy,
)

batch_move_tool = ToolDefinition(
    name="batch_move",
    description=(
        "Move or rename a file or directory inside the workspace. Fails if the destination exists. "
        "Do not use run_command for mv."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    },
    validator=_validate_pair,
    run=_run_move,
)

batch_delete_tool = ToolDefinition(
    name="batch_delete",
    description=(
        "Delete a file, or a directory when recursive=true, inside the workspace. "
        "Do not use run_command for rm."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean"},
        },
        "required": ["path"],
    },
    validator=_validate_delete,
    run=_run_delete,
)
