from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str


@dataclass(slots=True)
class ToolContext:
    """Runtime facts a tool needs: workspace root, preview, and optional approval."""

    cwd: str
    on_write_preview: Callable[[str, str], None] | None = None
    on_approve: Callable[[str], bool] | None = None

    def approve(self, summary: str) -> bool:
        """Return True when the user allows a write or command. None means auto-approve."""

        if self.on_approve is None:
            return True
        return bool(self.on_approve(summary))


Validator = Callable[[Any], Any]
Runner = Callable[[Any, ToolContext], ToolResult]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    validator: Validator
    run: Runner


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def find(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def execute(self, tool_name: str, input_data: Any, context: ToolContext) -> ToolResult:
        tool = self.find(tool_name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool: {tool_name}")
        try:
            parsed = tool.validator(input_data)
            return tool.run(parsed, context)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:  # noqa: BLE001
            return ToolResult(ok=False, output=f"{tool_name} error: {error}")
