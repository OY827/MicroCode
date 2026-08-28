from __future__ import annotations

from pathlib import Path

from microcode.types import ChatMessage

MEMORY_NAME = "MEMORY.md"
DEFAULT_MAX_CHARS = 8_000
MEMORY_HEADING = "## Project memory"


class ProjectMemory:
    """Workspace notes that persist across sessions and are injected into system."""

    def __init__(self, workspace: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.workspace = Path(workspace)
        self.path = self.workspace / ".microcode" / MEMORY_NAME
        self.max_chars = max_chars

    def read(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def display(self) -> str:
        text = self.read()
        if not text:
            return (
                "No project memory yet.\n"
                "Use /memory add <note> or edit .microcode/MEMORY.md"
            )
        return f"{self.path}\n\n{text}"

    def append(self, note: str) -> str:
        cleaned = note.strip()
        if not cleaned:
            return "Usage: /memory add <note>"
        bullet = cleaned if cleaned.startswith(("- ", "* ")) else f"- {cleaned}"
        existing = self.read()
        body = f"{existing}\n{bullet}\n" if existing else f"# Project memory\n\n{bullet}\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(body, encoding="utf-8")
        return f"Remembered: {bullet}"

    def inject(self, system_prompt: str) -> str:
        body = self.read()
        if not body:
            return system_prompt
        if len(body) > self.max_chars:
            body = body[: self.max_chars].rstrip() + "\n... (memory truncated)"
        suffix = (
            f"{MEMORY_HEADING}\n\n"
            "The following notes persist across sessions. "
            "Follow them unless the user says otherwise.\n\n"
            f"{body}"
        )
        return f"{system_prompt.rstrip()}\n\n{suffix}"


def apply_system_message(messages: list[ChatMessage], content: str) -> list[ChatMessage]:
    """Replace or insert the leading system message. Leaves other rows unchanged."""

    system = {"role": "system", "content": content}
    next_messages = list(messages)
    if next_messages and next_messages[0].get("role") == "system":
        next_messages[0] = system
        return next_messages
    return [system, *next_messages]


def parse_memory_command(text: str) -> tuple[str, str]:
    """Return ('show'|'add'|'help', payload) for a /memory line."""

    rest = text.strip()[len("/memory") :].strip()
    if not rest:
        return "show", ""
    lowered = rest.lower()
    if lowered == "add":
        return "add", ""
    if lowered.startswith("add "):
        return "add", rest[4:].strip()
    return "help", ""
