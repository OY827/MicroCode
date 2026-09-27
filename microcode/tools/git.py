from __future__ import annotations

import os
import subprocess
from pathlib import Path

from microcode.file_edit import PROTECTED_NAMES
from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import relative_workspace_path, resolve_tool_path

ACTIONS = ("status", "diff", "log", "add", "commit")
DEFAULT_MAX_LINES = 80
MAX_MAX_LINES = 200
GIT_TIMEOUT = 30


def _normalize_paths(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("paths must be a string or a list of strings")
    paths: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("each path must be a non-empty string")
        text = item.strip()
        if text.startswith("-"):
            raise ValueError("git add flags are not allowed; pass file paths only")
        paths.append(text)
    return paths


def _validate(input_data: dict) -> dict:
    action = input_data.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(ACTIONS)}")
    message = input_data.get("message", "")
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if action == "commit" and not message.strip():
        raise ValueError("message is required for commit")
    paths = _normalize_paths(input_data.get("paths", input_data.get("path")))
    if action == "add" and not paths:
        raise ValueError("paths is required for add")
    raw_lines = input_data.get("max_lines", DEFAULT_MAX_LINES)
    if isinstance(raw_lines, bool):
        raise ValueError("max_lines must be an integer")
    if isinstance(raw_lines, float) and raw_lines.is_integer():
        raw_lines = int(raw_lines)
    if not isinstance(raw_lines, int) or raw_lines < 1 or raw_lines > MAX_MAX_LINES:
        raise ValueError(f"max_lines must be between 1 and {MAX_MAX_LINES}")
    return {
        "action": action,
        "message": message.strip(),
        "max_lines": raw_lines,
        "paths": paths,
    }


def _run_git(args: list[str], cwd: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n... ({omitted} more lines omitted)"


def _run_status(cwd: str) -> ToolResult:
    code, stdout, stderr = _run_git(["status", "--short"], cwd)
    if code != 0:
        return ToolResult(ok=False, output=stderr or "git status failed")
    if not stdout:
        return ToolResult(ok=True, output="Working tree clean.")
    return ToolResult(ok=True, output=stdout)


def _run_diff(cwd: str, max_lines: int) -> ToolResult:
    unstaged_code, unstaged, unstaged_err = _run_git(["diff"], cwd)
    if unstaged_code != 0:
        return ToolResult(ok=False, output=unstaged_err or "git diff failed")
    staged_code, staged, staged_err = _run_git(["diff", "--cached"], cwd)
    if staged_code != 0:
        return ToolResult(ok=False, output=staged_err or "git diff --cached failed")
    parts = []
    if staged:
        parts.append("Staged:\n" + staged)
    if unstaged:
        parts.append("Unstaged:\n" + unstaged)
    if not parts:
        return ToolResult(ok=True, output="No unstaged or staged changes.")
    return ToolResult(ok=True, output=_truncate("\n\n".join(parts), max_lines))


def _run_log(cwd: str, max_lines: int) -> ToolResult:
    code, stdout, stderr = _run_git(["log", "--oneline", f"-{max_lines}"], cwd)
    if code != 0:
        return ToolResult(ok=False, output=stderr or "git log failed")
    if not stdout:
        return ToolResult(ok=True, output="No commits.")
    return ToolResult(ok=True, output=stdout)


def _protected_under(path: Path) -> Path | None:
    if path.name in PROTECTED_NAMES:
        return path
    if path.is_file() or not path.exists():
        return None
    for dirpath, dirnames, filenames in os.walk(path):
        for name in list(dirnames) + filenames:
            if name in PROTECTED_NAMES:
                return Path(dirpath) / name
    return None


def _run_add(paths: list[str], context) -> ToolResult:
    rels: list[str] = []
    for path in paths:
        try:
            target = resolve_tool_path(context, path)
        except PermissionError as error:
            return ToolResult(ok=False, output=str(error))
        if not target.exists():
            return ToolResult(ok=False, output=f"Path not found: {path}")
        protected = _protected_under(target)
        if protected is not None:
            label = relative_workspace_path(context.cwd, protected)
            return ToolResult(ok=False, output=f"Refusing to stage protected file: {label}")
        rels.append(relative_workspace_path(context.cwd, target))

    command = "git add " + " ".join(rels)
    if not context.approve(f"Stage files?\n" + "\n".join(rels), kind="command", key=command):
        return ToolResult(ok=False, output=context.reject_text(f"command: {command}"))

    code, stdout, stderr = _run_git(["add", "--", *rels], context.cwd)
    if code != 0:
        return ToolResult(ok=False, output=stderr or "git add failed")
    status = _run_status(context.cwd)
    staged = f"Staged {len(rels)} path(s): {', '.join(rels)}"
    if stdout:
        staged = stdout + "\n" + staged
    if status.ok and status.output:
        return ToolResult(ok=True, output=f"{staged}\n{status.output}")
    return ToolResult(ok=True, output=staged)


def _run_commit(cwd: str, message: str, context) -> ToolResult:
    staged_code, _, staged_err = _run_git(["diff", "--cached", "--quiet"], cwd)
    if staged_code == 0:
        return ToolResult(ok=False, output="No staged changes. Stage files before commit.")
    if staged_code not in {0, 1}:
        return ToolResult(ok=False, output=staged_err or "git diff --cached failed")

    command = f"git commit -m {message}"
    if not context.approve(f"Create git commit?\n{message}", kind="command", key=command):
        return ToolResult(ok=False, output=context.reject_text(f"command: {command}"))

    code, stdout, stderr = _run_git(["commit", "-m", message], cwd)
    if code != 0:
        return ToolResult(ok=False, output=stderr or "git commit failed")
    return ToolResult(ok=True, output=stdout or f"Committed: {message}")


def _run(input_data: dict, context) -> ToolResult:
    try:
        action = input_data["action"]
        if action == "status":
            return _run_status(context.cwd)
        if action == "diff":
            return _run_diff(context.cwd, input_data["max_lines"])
        if action == "log":
            return _run_log(context.cwd, input_data["max_lines"])
        if action == "add":
            return _run_add(input_data["paths"], context)
        return _run_commit(context.cwd, input_data["message"], context)
    except FileNotFoundError:
        return ToolResult(ok=False, output="git is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output="git command timed out.")


git_tool = ToolDefinition(
    name="git",
    description=(
        "Inspect, stage, or commit the workspace git repo. "
        "status and log are read-only; diff shows staged and unstaged patches; "
        "add stages listed paths (no flags); commit requires staged files and user approval. "
        "Use run_command for other git verbs such as push."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "message": {"type": "string", "description": "Commit message, required for commit."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File or folder paths to stage, required for add.",
            },
            "path": {"type": "string", "description": "Single path to stage (add)."},
            "max_lines": {"type": "integer"},
        },
        "required": ["action"],
    },
    validator=_validate,
    run=_run,
)
