from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["pending", "in_progress", "completed"]
STATUSES: tuple[Status, ...] = ("pending", "in_progress", "completed")
MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


@dataclass(slots=True)
class TodoItem:
    content: str
    status: Status = "pending"


@dataclass
class TodoStore:
    """In-memory checklist for the current process. Not written to disk."""

    items: list[TodoItem] = field(default_factory=list)

    def replace(self, todos: list[TodoItem]) -> str:
        self.items = list(todos)
        return self.format()

    def format(self) -> str:
        if not self.items:
            return "No todos."
        counts = {status: 0 for status in STATUSES}
        lines = ["todos:"]
        for item in self.items:
            counts[item.status] += 1
            lines.append(f"  {MARKERS[item.status]} {item.content}")
        lines.append(
            f"{counts['pending']} pending, {counts['in_progress']} in progress, "
            f"{counts['completed']} completed"
        )
        return "\n".join(lines)
