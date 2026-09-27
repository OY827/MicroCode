from __future__ import annotations

from pathlib import Path

from microcode.tooling import ToolDefinition, ToolResult
from microcode.tools.list_files import SKIP_NAMES
from microcode.workspace import resolve_tool_path

TREE_SKIP_NAMES = SKIP_NAMES | {"node_modules", "dist", "build", ".tox"}
DEFAULT_DEPTH = 3
MAX_DEPTH = 8
MAX_LINES = 400


def _validate(input_data: dict) -> dict:
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    raw_depth = input_data.get("max_depth", DEFAULT_DEPTH)
    if isinstance(raw_depth, bool):
        raise ValueError("max_depth must be an integer")
    if isinstance(raw_depth, float) and raw_depth.is_integer():
        raw_depth = int(raw_depth)
    if not isinstance(raw_depth, int):
        raise ValueError("max_depth must be an integer")
    if raw_depth < 1 or raw_depth > MAX_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
    return {"path": input_data.get("path", "."), "max_depth": raw_depth}


def _visible(path: Path) -> list[Path]:
    try:
        entries = [entry for entry in path.iterdir() if entry.name not in TREE_SKIP_NAMES]
    except OSError:
        return []
    return sorted(entries, key=lambda item: (not item.is_dir(), item.name.lower()))


def _walk(path: Path, prefix: str, remaining_depth: int, lines: list[str]) -> None:
    entries = _visible(path)
    for index, entry in enumerate(entries):
        if len(lines) >= MAX_LINES:
            lines.append(f"{prefix}...")
            return
        last = index == len(entries) - 1
        connector = "└── " if last else "├── "
        child_prefix = f"{prefix}{'    ' if last else '│   '}"
        kind = "dir " if entry.is_dir() else "file"
        lines.append(f"{prefix}{connector}{kind} {entry.name}")
        if not entry.is_dir():
            continue
        if remaining_depth > 1:
            _walk(entry, child_prefix, remaining_depth - 1, lines)
        elif _visible(entry):
            if len(lines) < MAX_LINES:
                lines.append(f"{child_prefix}...")


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    if target.is_file():
        return ToolResult(ok=True, output=f"file {Path(input_data['path']).name}")

    root = Path(input_data["path"]).as_posix() if input_data["path"] not in {"", "."} else "."
    lines = [root]
    _walk(target, "", input_data["max_depth"], lines)
    if len(lines) == 1:
        return ToolResult(ok=True, output=f"{root}\n(empty)")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + ["..."]
    return ToolResult(ok=True, output="\n".join(lines))


file_tree_tool = ToolDefinition(
    name="file_tree",
    description=(
        "Show a limited-depth directory tree. Use this to see structure; "
        "use list_files for a single folder and read_file for contents."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_depth": {
                "type": "integer",
                "description": f"Directory levels to expand (default {DEFAULT_DEPTH}, max {MAX_DEPTH}).",
            },
        },
    },
    validator=_validate,
    run=_run,
)
