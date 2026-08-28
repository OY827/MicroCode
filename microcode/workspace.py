from __future__ import annotations

from pathlib import Path

from microcode.tooling import ToolContext


def workspace_root(cwd: str) -> Path:
    return Path(cwd).resolve()


def resolve_tool_path(context: ToolContext, input_path: str) -> Path:
    """Resolve a tool path and refuse anything that escapes the workspace."""

    candidate = Path(input_path)
    target = candidate if candidate.is_absolute() else Path(context.cwd) / candidate
    normalized = target.resolve()
    try:
        normalized.relative_to(workspace_root(context.cwd))
    except ValueError:
        raise PermissionError(f"Path escapes workspace: {input_path}")
    return normalized


def relative_workspace_path(cwd: str, target: Path) -> str:
    """Stable posix path relative to the workspace, used as a checkpoint key."""

    return target.resolve().relative_to(workspace_root(cwd)).as_posix()
