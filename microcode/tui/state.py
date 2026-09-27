from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from microcode.jobs import JobStore
from microcode.memory import ProjectMemory
from microcode.permissions import PermissionStore
from microcode.session import Session, SessionStore
from microcode.todos import TodoStore
from microcode.tooling import ToolRegistry
from microcode.tui.types import TranscriptEntry
from microcode.types import ChatMessage, ModelAdapter
from microcode.usage import UsageLedger


@dataclass
class PendingPrompt:
    kind: Literal["approve", "ask", "edit"]
    summary: str
    answer: str = ""
    proposed: str = ""


@dataclass
class TuiContext:
    model: ModelAdapter
    tools: ToolRegistry
    messages: list[ChatMessage]
    cwd: str
    session: Session | None = None
    store: SessionStore | None = None
    memory: ProjectMemory | None = None
    system_prompt: str | None = None
    mcp_status: str | None = None
    permissions: PermissionStore | None = None
    todos: TodoStore | None = None
    jobs: JobStore | None = None
    usage: UsageLedger | None = None


@dataclass
class ScreenState:
    input: str = ""
    cursor_offset: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    transcript_scroll_offset: int = 0
    selected_slash_index: int = 0
    status: str | None = None
    active_tool: str | None = None
    history: list[str] = field(default_factory=list)
    history_index: int = 0
    history_draft: str = ""
    next_entry_id: int = 1
    pending: PendingPrompt | None = None
    is_busy: bool = False
    stream_entry_id: int | None = None
    agent_done: bool = False
    agent_error: str | None = None
    should_exit: bool = False
    resolve_approval: Callable[[str], None] | None = None
    resolve_ask: Callable[[str], None] | None = None

    def push(
        self,
        kind: str,
        body: str,
        *,
        tool_name: str | None = None,
        status: str | None = None,
        collapsed: bool = False,
        collapsed_summary: str | None = None,
    ) -> int:
        entry = TranscriptEntry(
            id=self.next_entry_id,
            kind=kind,  # type: ignore[arg-type]
            body=body,
            tool_name=tool_name,
            status=status,  # type: ignore[arg-type]
            collapsed=collapsed,
            collapsed_summary=collapsed_summary,
        )
        self.transcript.append(entry)
        self.next_entry_id += 1
        self.transcript_scroll_offset = 0
        return entry.id

    def find(self, entry_id: int) -> TranscriptEntry | None:
        for entry in self.transcript:
            if entry.id == entry_id:
                return entry
        return None
