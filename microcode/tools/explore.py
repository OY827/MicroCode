from __future__ import annotations

from microcode.tooling import ToolDefinition, ToolResult

EXPLORE_PROMPT = (
    "You are an Explore sub-agent. Search the workspace with list_files, grep_files, "
    "and read_file only. Do not edit files, run commands, or ask the user. "
    "When you have enough, return a concise summary of what you found: key paths, "
    "how they relate, and anything the parent agent should read next."
)
MAX_EXPLORE_STEPS = 8


def _validate(input_data: dict) -> dict:
    task = input_data.get("task") or input_data.get("prompt")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task is required")
    return {"task": task.strip()}


def _run(input_data: dict, context) -> ToolResult:
    if context.model is None:
        return ToolResult(ok=False, output="No model available to run explore.")

    from microcode.agent_loop import run_agent_turn
    from microcode.tools import create_explore_tool_registry

    def on_tool_start(name: str, args: dict) -> None:
        if context.on_tool_start:
            context.on_tool_start(f"explore/{name}", args)

    def on_tool_result(name: str, output: str, is_error: bool) -> None:
        if context.on_tool_result:
            context.on_tool_result(f"explore/{name}", output, is_error)

    nested = run_agent_turn(
        model=context.model,
        tools=create_explore_tool_registry(),
        messages=[
            {"role": "system", "content": EXPLORE_PROMPT},
            {"role": "user", "content": input_data["task"]},
        ],
        cwd=context.cwd,
        max_steps=MAX_EXPLORE_STEPS,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
    )
    summary = _last_assistant(nested)
    if not summary:
        return ToolResult(ok=False, output="Explore finished without a summary.")
    return ToolResult(ok=True, output=summary)


def _last_assistant(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


explore_tool = ToolDefinition(
    name="explore",
    description=(
        "Spawn a read-only sub-agent to search the workspace. It can list, grep, and read "
        "files, then returns a summary. Use this for broad codebase questions so the main "
        "conversation does not fill up with raw file contents."
    ),
    input_schema={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    validator=_validate,
    run=_run,
)
