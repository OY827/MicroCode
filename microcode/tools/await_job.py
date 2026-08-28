from __future__ import annotations

from microcode.jobs import jobs_from
from microcode.tooling import ToolDefinition, ToolResult

DEFAULT_WAIT = 120
MAX_WAIT = 120


def _validate(input_data: dict) -> dict:
    job_id = input_data.get("job_id") or input_data.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id is required")
    timeout = int(input_data.get("timeout", DEFAULT_WAIT))
    if timeout < 1:
        raise ValueError("timeout must be >= 1")
    return {"job_id": job_id.strip(), "timeout": min(timeout, MAX_WAIT)}


def _run(input_data: dict, context) -> ToolResult:
    job = jobs_from(context).get(input_data["job_id"])
    if job is None:
        return ToolResult(ok=False, output=f"Unknown job: {input_data['job_id']}")
    finished, text = job.collect(input_data["timeout"])
    if not finished:
        return ToolResult(ok=False, output=text)
    return ToolResult(ok=job.returncode == 0, output=text)


await_job_tool = ToolDefinition(
    name="await_job",
    description=(
        "Wait for a background job started by run_command(background=true) and return "
        "its stdout/stderr/exit code. Pass the job_id from the start result."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["job_id"],
    },
    validator=_validate,
    run=_run,
)
