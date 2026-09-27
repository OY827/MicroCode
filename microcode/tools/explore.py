from __future__ import annotations

from microcode.tooling import ToolDefinition, ToolResult
from microcode.tools.task import run_subagent


def _validate(input_data: dict) -> dict:
    task = input_data.get("task") or input_data.get("prompt")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task is required")
    return {"task": task.strip()}


def _run(input_data: dict, context) -> ToolResult:
    return run_subagent("explore", input_data["task"], context)


explore_tool = ToolDefinition(
    name="explore",
    description=(
        "Spawn a read-only sub-agent to search the workspace. It can list, tree, grep, "
        "find Python symbols, and read files, then returns a summary. Prefer task with "
        "files, then returns a summary. Prefer task with agent_type=explore for new calls; "
        "this name is kept for compatibility."
    ),
    input_schema={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    validator=_validate,
    run=_run,
)
