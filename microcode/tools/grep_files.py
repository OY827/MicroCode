from __future__ import annotations

import re
from pathlib import Path

from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import resolve_tool_path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    ".egg-info",
}
SKIP_FILES = {".env"}
MAX_FILES = 2000
MAX_MATCHES = 200


def _validate(input_data: dict) -> dict:
    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern is required")
    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    return {"pattern": pattern, "path": path}


def _should_skip(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return True
    return any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts)


def _run(input_data: dict, context) -> ToolResult:
    root = resolve_tool_path(context, input_data["path"])
    try:
        regex = re.compile(input_data["pattern"])
    except re.error as error:
        return ToolResult(ok=False, output=f"Invalid regex: {error}")

    if root.is_file():
        files = [root]
    elif root.is_dir():
        try:
            files = sorted(path for path in root.rglob("*") if path.is_file())
        except OSError as error:
            return ToolResult(ok=False, output=str(error))
    else:
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")

    matches: list[str] = []
    scanned = 0
    skipped = 0
    workspace = Path(context.cwd).resolve()

    for file_path in files:
        if _should_skip(file_path):
            skipped += 1
            continue
        if scanned >= MAX_FILES:
            break
        scanned += 1
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue
        relative = file_path.resolve().relative_to(workspace).as_posix()
        for index, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{relative}:{index}:{line}")
                if len(matches) >= MAX_MATCHES:
                    extra = f"\n\nResults truncated at {MAX_MATCHES} matches."
                    return ToolResult(ok=True, output="\n".join(matches) + extra)

    output = "\n".join(matches) if matches else "No matches found."
    if skipped:
        output += f"\n({skipped} file(s) skipped)"
    return ToolResult(ok=True, output=output)


grep_files_tool = ToolDefinition(
    name="grep_files",
    description="Search UTF-8 text files under the workspace using a regex pattern.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    },
    validator=_validate,
    run=_run,
)
