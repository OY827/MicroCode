from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from microcode.checkpoint import WriteCheckpoint
    from microcode.jobs import JobStore
    from microcode.permissions import Kind, PermissionStore
    from microcode.todos import TodoStore
    from microcode.types import ModelAdapter


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str


@dataclass(slots=True)
class ToolContext:
    """Runtime facts a tool needs: workspace root, preview, and optional approval."""

    cwd: str
    on_write_preview: Callable[[str, str], None] | None = None
    on_approve: Callable[[str], Any] | None = None
    checkpoint: WriteCheckpoint | None = None
    permissions: PermissionStore | None = None
    todos: TodoStore | None = None
    jobs: JobStore | None = None
    on_ask_user: Callable[[str], str] | None = None
    model: ModelAdapter | None = None
    on_tool_start: Callable[[str, dict], None] | None = None
    on_tool_result: Callable[[str, str, bool], None] | None = None

    def approve(self, summary: str, *, kind: Kind | None = None, key: str | None = None) -> bool:
        """Return True when the user allows a write or command. None means auto-approve."""

        from microcode.permissions import parse_decision

        if self.permissions is not None and kind and key and self.permissions.is_allowed(kind, key):
            return True
        if self.on_approve is None:
            return True
        allowed, remember = parse_decision(self.on_approve(summary))
        if allowed and remember and self.permissions is not None and kind and key:
            if self.permissions.allow(kind, key):
                self.permissions.save()
        return allowed

    def ask(self, question: str) -> str | None:
        """Return the user's one-line answer, or None when nobody can be asked."""

        if self.on_ask_user is None:
            return None
        return self.on_ask_user(question)


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

    def add(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

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
