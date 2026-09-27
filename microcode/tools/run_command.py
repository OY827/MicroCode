from __future__ import annotations

import subprocess

from microcode.command_guard import argv_for, refuse_command
from microcode.jobs import format_command_output, jobs_from
from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import resolve_tool_path

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120


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
    background = input_data.get("background", False)
    if isinstance(background, str):
        background = background.strip().lower() in {"1", "true", "yes"}
    else:
        background = bool(background)
    return {
        "command": command.strip(),
        "timeout": min(timeout, MAX_TIMEOUT),
        "cwd": cwd,
        "background": background,
    }


def _run(input_data: dict, context) -> ToolResult:
    blocked = refuse_command(input_data["command"])
    if blocked:
        return ToolResult(ok=False, output=blocked)

    workdir = context.cwd
    if input_data.get("cwd"):
        workdir = str(resolve_tool_path(context, input_data["cwd"]))

    argv = argv_for(input_data["command"])
    kind = "background command" if input_data["background"] else "command"
    summary = (
        f"Run {kind} in {workdir} (timeout {input_data['timeout']}s):\n"
        f"{input_data['command']}"
    )
    if not context.approve(summary, kind="command", key=input_data["command"]):
        return ToolResult(ok=False, output=context.reject_text(f"command: {input_data['command']}"))

    if input_data["background"]:
        job = jobs_from(context).start(input_data["command"], workdir, argv=argv)
        return ToolResult(
            ok=True,
            output=(
                f"started {job.id}\n"
                f"command: {job.command}\n"
                "Use await_job with this id to collect stdout/stderr."
            ),
        )

    try:
        completed = subprocess.run(
            argv,
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

    output = format_command_output(
        completed.stdout.strip(), completed.stderr.strip(), completed.returncode
    )
    return ToolResult(ok=completed.returncode == 0, output=output)


run_command_tool = ToolDefinition(
    name="run_command",
    description=(
        "Run one program in the workspace (no system shell, no pipes or &&). "
        "Use this to run tests or scripts after editing files. Default timeout is 30 seconds. "
        "Set background=true to start the command and return a job id immediately; "
        "collect output later with await_job. "
        "Destructive git/disk commands are refused."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number"},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
    },
    validator=_validate,
    run=_run,
)
