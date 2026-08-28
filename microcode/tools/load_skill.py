from __future__ import annotations

from microcode.skills import load_skill
from microcode.tooling import ToolDefinition, ToolResult


def _validate(input_data: dict) -> dict:
    name = input_data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    return {"name": name.strip()}


def _run(input_data: dict, context) -> ToolResult:
    skill = load_skill(context.cwd, input_data["name"])
    if skill is None:
        return ToolResult(ok=False, output=f"Unknown skill: {input_data['name']}")
    return ToolResult(
        ok=True,
        output="\n".join(
            [
                f"SKILL: {skill.name}",
                f"PATH: {skill.path}",
                "",
                skill.content,
            ]
        ),
    )


load_skill_tool = ToolDefinition(
    name="load_skill",
    description=(
        "Load a local SKILL.md by name from .microcode/skills/<name>/SKILL.md. "
        "Use when a listed skill matches the current task."
    ),
    input_schema={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    validator=_validate,
    run=_run,
)
