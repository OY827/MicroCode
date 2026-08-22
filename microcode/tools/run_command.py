from __future__ import annotations

import subprocess

from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import resolve_tool_path

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_OUTPUT_CHARS = 8000


def _validate(input_data: dict) -> dict:
    command = input_data.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command is required")
    timeout = int(input_data.get("timeout", DEFAULT_TIMEOUT))
    if timeout < 1:
        raise ValueError("timeout must be >= 1")
    cwd = input_data.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("cwd must be a string")
    return {
        "command": command.strip(),
        "timeout": min(timeout, MAX_TIMEOUT),
        "cwd": cwd,
    }


def _format_output(stdout: str, stderr: str, returncode: int) -> str:
    parts = [f"exit_code: {returncode}"]
    if stdout:
        parts.extend(["", "stdout:", stdout])
    if stderr:
        parts.extend(["", "stderr:", stderr])
    text = "\n".join(parts).strip()
    if len(text) > MAX_OUTPUT_CHARS:
        omitted = len(text) - MAX_OUTPUT_CHARS
        text = text[:MAX_OUTPUT_CHARS] + f"\n... ({omitted} more chars omitted)"
    return text


def _run(input_data: dict, context) -> ToolResult:
    workdir = context.cwd
    if input_data.get("cwd"):
        workdir = str(resolve_tool_path(context, input_data["cwd"]))

    summary = (
        f"Run command in {workdir} (timeout {input_data['timeout']}s):\n"
        f"{input_data['command']}"
    )
    if not context.approve(summary):
        return ToolResult(ok=False, output=f"User rejected command: {input_data['command']}")

    try:
        completed = subprocess.run(
            input_data["command"],
            shell=True,
            cwd=workdir,
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
            output=(
                f"Command timed out after {input_data['timeout']} seconds: "
                f"{input_data['command']}"
            ),
        )
    except OSError as error:
        return ToolResult(ok=False, output=str(error))

    output = _format_output(completed.stdout.strip(), completed.stderr.strip(), completed.returncode)
    return ToolResult(ok=completed.returncode == 0, output=output)


run_command_tool = ToolDefinition(
    name="run_command",
    description=(
        "Run a shell command in the workspace directory and return stdout/stderr. "
        "Use this to run tests or scripts after editing files. Default timeout is 30 seconds."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
    },
    validator=_validate,
    run=_run,
)
