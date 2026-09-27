from __future__ import annotations

from typing import TypedDict

from microcode.tooling import ToolDefinition, ToolResult

READ_ONLY_TOOLS = frozenset(
    {"list_files", "file_tree", "read_file", "grep_files", "find_symbols", "find_references"}
)
NESTED_AGENT_TOOLS = frozenset({"task", "explore"})


class AgentDef(TypedDict):
    label: str
    system_prompt: str
    allowed_tools: frozenset[str] | None
    max_steps: int


AGENT_TYPES: dict[str, AgentDef] = {
    "explore": {
        "label": "Explore",
        "system_prompt": (
            "You are an Explore sub-agent. Search the workspace with list_files, file_tree, "
            "grep_files, find_symbols, find_references, and read_file only. Do not edit files, run commands, or ask the user. "
            "When you have enough, return a concise summary of what you found: key paths, "
            "how they relate, and anything the parent agent should read next."
        ),
        "allowed_tools": READ_ONLY_TOOLS,
        "max_steps": 8,
    },
    "plan": {
        "label": "Plan",
        "system_prompt": (
            "You are a Plan sub-agent. Read the workspace thoroughly with list_files, "
            "file_tree, grep_files, find_symbols, find_references, and read_file only. Trace the code paths that matter for the task. "
            "Do not edit files, run commands, or ask the user. "
            "When done, return a detailed analysis and a concrete recommended plan."
        ),
        "allowed_tools": READ_ONLY_TOOLS,
        "max_steps": 8,
    },
    "general": {
        "label": "General",
        "system_prompt": (
            "You are a General sub-agent. You can read and change the workspace. "
            "Do not spawn another sub-agent. Work autonomously; do not ask the user. "
            "When done, return a concise summary of what you changed and what still needs attention."
        ),
        "allowed_tools": None,
        "max_steps": 15,
    },
}


def _validate(input_data: dict) -> dict:
    agent_type = str(input_data.get("agent_type") or "general").strip()
    if agent_type not in AGENT_TYPES:
        valid = ", ".join(AGENT_TYPES)
        raise ValueError(f"agent_type must be one of: {valid}")
    prompt = input_data.get("prompt") or input_data.get("task") or input_data.get("description")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        description = prompt.strip()
    return {
        "description": description.strip(),
        "prompt": prompt.strip(),
        "agent_type": agent_type,
    }


def run_subagent(agent_type: str, prompt: str, context) -> ToolResult:
    """Run an isolated agent loop and return only the final summary to the parent."""

    if context.model is None:
        return ToolResult(ok=False, output="No model available to run a sub-agent.")

    from microcode.agent_loop import run_agent_turn
    from microcode.tools import create_subagent_tool_registry

    spec = AGENT_TYPES[agent_type]
    prefix = agent_type
    read_only = spec["allowed_tools"] is not None

    def on_tool_start(name: str, args: dict) -> None:
        if context.on_tool_start:
            context.on_tool_start(f"{prefix}/{name}", args)

    def on_tool_result(name: str, output: str, is_error: bool) -> None:
        if context.on_tool_result:
            context.on_tool_result(f"{prefix}/{name}", output, is_error)

    nested_tape = context.tape.spawn() if context.tape is not None else None
    if nested_tape is not None:
        nested_tape.record_user(prompt)
    nested = run_agent_turn(
        model=context.model,
        tools=create_subagent_tool_registry(spec["allowed_tools"]),
        messages=[
            {"role": "system", "content": spec["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        cwd=context.cwd,
        max_steps=spec["max_steps"],
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_write_preview=None if read_only else context.on_write_preview,
        on_approve=None if read_only else context.on_approve,
        on_revise_write=None if read_only else context.on_revise_write,
        on_ask_user=None,
        checkpoint=None if read_only else context.checkpoint,
        permissions=None if read_only else context.permissions,
        todos=None if read_only else context.todos,
        jobs=None if read_only else context.jobs,
        permission_mode="read" if read_only else getattr(context, "permission_mode", "ask"),
        tape=nested_tape,
    )
    summary = _last_assistant(nested)
    if not summary:
        return ToolResult(ok=False, output=f"{spec['label']} finished without a summary.")
    return ToolResult(ok=True, output=summary)


def _last_assistant(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


def _run(input_data: dict, context) -> ToolResult:
    return run_subagent(input_data["agent_type"], input_data["prompt"], context)


task_tool = ToolDefinition(
    name="task",
    description=(
        "Spawn an isolated sub-agent and get back only its final summary. "
        "Use explore for a fast read-only search, plan for a thorough read-only analysis, "
        "or general for a multi-step change that should not fill the parent conversation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short label for the sub-task.",
            },
            "prompt": {
                "type": "string",
                "description": "Full instructions for the sub-agent. Defaults to description.",
            },
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "general"],
                "description": "explore and plan are read-only; general may edit the workspace.",
            },
        },
        "required": ["description"],
    },
    validator=_validate,
    run=_run,
)
