from __future__ import annotations

from microcode.todos import STATUSES, TodoItem, TodoStore
from microcode.tooling import ToolDefinition, ToolResult


def _validate(input_data: dict) -> dict:
    raw = input_data.get("todos")
    if not isinstance(raw, list):
        raise ValueError("todos must be a list")
    parsed: list[TodoItem] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"todos[{index}] must be an object")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"todos[{index}].content is required")
        status = item.get("status", "pending")
        if status not in STATUSES:
            raise ValueError(
                f"todos[{index}].status must be pending, in_progress, or completed"
            )
        parsed.append(TodoItem(content=content.strip(), status=status))
    return {"todos": parsed}


def _run(input_data: dict, context) -> ToolResult:
    store = context.todos
    if store is None:
        store = TodoStore()
        context.todos = store
    return ToolResult(ok=True, output=store.replace(input_data["todos"]))


todo_write_tool = ToolDefinition(
    name="todo_write",
    description=(
        "Replace the in-memory task list for this session. Use it for multi-step work. "
        "Pass the complete list each time. Each item needs content and optional status "
        "(pending, in_progress, completed)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["todos"],
    },
    validator=_validate,
    run=_run,
)
