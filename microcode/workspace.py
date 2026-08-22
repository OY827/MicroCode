from __future__ import annotations

from pathlib import Path

from microcode.tooling import ToolContext


def resolve_tool_path(context: ToolContext, input_path: str) -> Path:
    """Resolve a tool path and refuse anything that escapes the workspace."""

    candidate = Path(input_path)
    target = candidate if candidate.is_absolute() else Path(context.cwd) / candidate
    normalized = target.resolve()
    workspace_root = Path(context.cwd).resolve()
    try:
        normalized.relative_to(workspace_root)
    except ValueError:
        raise PermissionError(f"Path escapes workspace: {input_path}")
    return normalized
