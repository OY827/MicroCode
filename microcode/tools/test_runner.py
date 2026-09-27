from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from microcode.tooling import ToolDefinition, ToolResult
from microcode.tools.list_files import SKIP_NAMES
from microcode.workspace import relative_workspace_path, resolve_tool_path

FRAMEWORKS = ("auto", "pytest", "unittest")
SKIP_DIRS = SKIP_NAMES | {"node_modules", "dist", "build", ".tox"}
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 180
MAX_OUTPUT_CHARS = 6000


def _as_int(value: object, default: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError("timeout must be an integer")
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if not isinstance(raw, int):
        raise ValueError("timeout must be an integer")
    return raw


def _validate(input_data: dict) -> dict:
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    framework = input_data.get("framework", "auto")
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be one of: {', '.join(FRAMEWORKS)}")
    pattern = input_data.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        raise ValueError("pattern must be a string")
    timeout = _as_int(input_data.get("timeout"), DEFAULT_TIMEOUT)
    if timeout < 1 or timeout > MAX_TIMEOUT:
        raise ValueError(f"timeout must be between 1 and {MAX_TIMEOUT}")
    return {
        "path": input_data.get("path", "."),
        "framework": framework,
        "pattern": (pattern or "").strip(),
        "timeout": timeout,
    }


def _is_test_file(path: Path) -> bool:
    name = path.name
    return path.suffix == ".py" and (name.startswith("test_") or name.endswith("_test.py"))


def discover_test_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _is_test_file(root) else []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            candidate = Path(dirpath) / name
            if _is_test_file(candidate):
                found.append(candidate)
    return sorted(found)


def _pytest_available() -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def resolve_framework(name: str) -> str:
    if name == "auto":
        return "pytest" if _pytest_available() else "unittest"
    return name


def build_test_command(framework: str, target: Path, *, pattern: str) -> list[str]:
    if framework == "pytest":
        command = [sys.executable, "-m", "pytest", str(target), "-q", "--tb=short"]
        if pattern:
            command.extend(["-k", pattern])
        return command
    if target.is_file():
        command = [sys.executable, "-m", "unittest", str(target)]
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-s", str(target), "-p", "test_*.py"]
    if pattern:
        command.extend(["-k", pattern])
    return command


def parse_pytest_summary(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key in counts:
        match = re.search(rf"(\d+) {key}", output)
        if match:
            counts[key] = int(match.group(1))
    error_match = re.search(r"(\d+) error", output)
    if error_match:
        counts["errors"] = int(error_match.group(1))
    return counts


def failed_test_names(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            names.append(line.split(" ", 1)[1].strip())
    return names


def _format_report(
    *,
    framework: str,
    path: str,
    files: list[Path],
    cwd: str,
    counts: dict[str, int],
    failed: list[str],
    output: str,
    returncode: int,
) -> str:
    rel_files = []
    for item in files[:20]:
        try:
            rel_files.append(relative_workspace_path(cwd, item))
        except ValueError:
            rel_files.append(item.name)
    lines = [
        f"framework: {framework}",
        f"path: {path}",
        f"files: {len(files)}",
        f"exit_code: {returncode}",
        f"passed: {counts['passed']}",
        f"failed: {counts['failed']}",
        f"errors: {counts['errors']}",
        f"skipped: {counts['skipped']}",
    ]
    if rel_files:
        lines.append("discovered:")
        lines.extend(f"  {name}" for name in rel_files)
        if len(files) > 20:
            lines.append(f"  ... ({len(files) - 20} more)")
    if failed:
        lines.append("failures:")
        lines.extend(f"  {name}" for name in failed[:20])
    clipped = output.strip()
    if len(clipped) > MAX_OUTPUT_CHARS:
        clipped = clipped[:MAX_OUTPUT_CHARS] + f"\n... ({len(output) - MAX_OUTPUT_CHARS} more chars omitted)"
    if clipped:
        lines.extend(["", "output:", clipped])
    return "\n".join(lines)


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")

    files = discover_test_files(target)
    if not files:
        return ToolResult(
            ok=False,
            output=f"No test files found in {input_data['path']} (expected test_*.py or *_test.py).",
        )

    framework = resolve_framework(input_data["framework"])
    command = build_test_command(framework, target, pattern=input_data["pattern"])
    summary = "Run tests:\n" + " ".join(command)
    if not context.approve(summary, kind="command", key=" ".join(command)):
        return ToolResult(ok=False, output=context.reject_text("tests: " + " ".join(command)))

    try:
        completed = subprocess.run(
            command,
            cwd=context.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=input_data["timeout"],
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            output=f"Tests timed out after {input_data['timeout']} seconds.",
        )
    except OSError as error:
        return ToolResult(ok=False, output=str(error))

    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    counts = parse_pytest_summary(combined)
    report = _format_report(
        framework=framework,
        path=input_data["path"],
        files=files,
        cwd=context.cwd,
        counts=counts,
        failed=failed_test_names(combined),
        output=combined,
        returncode=completed.returncode,
    )
    return ToolResult(ok=completed.returncode == 0, output=report)


test_runner_tool = ToolDefinition(
    name="test_runner",
    description=(
        "Discover and run Python tests in the workspace (pytest if available, else unittest). "
        "Returns pass/fail counts and failure names. Use this after code changes instead of "
        "guessing a pytest command. Use run_command for non-test scripts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Test file or directory (default: workspace root)."},
            "framework": {"type": "string", "enum": list(FRAMEWORKS)},
            "pattern": {"type": "string", "description": "pytest/unittest -k expression."},
            "timeout": {"type": "integer"},
        },
    },
    validator=_validate,
    run=_run,
)
