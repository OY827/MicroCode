from __future__ import annotations

from microcode.compact import compact_limit, context_chars
from microcode.memory import ProjectMemory
from microcode.models import model_lines
from microcode.permissions import PermissionStore
from microcode.session import Session
from microcode.types import ChatMessage, ModelAdapter
from microcode.usage import UsageLedger


def format_status(
    *,
    cwd: str,
    messages: list[ChatMessage],
    session: Session | None = None,
    memory: ProjectMemory | None = None,
    permissions: PermissionStore | None = None,
    model: ModelAdapter | None = None,
    usage: UsageLedger | None = None,
) -> str:
    """Plain snapshot of this chat. Local only."""

    user_turns = sum(1 for message in messages if message.get("role") == "user")
    used = context_chars(messages)
    limit = compact_limit()
    notes = len(memory.manager.all_entries()) if memory is not None else 0
    writes = len(permissions.writes) if permissions is not None else 0
    commands = len(permissions.commands) if permissions is not None else 0
    undoable = len(session.checkpoint) if session is not None else 0
    restore_points = len(session.checkpoints) if session is not None else 0
    redoable = len(session.undone) if session is not None else 0
    mode = session.permission_mode if session is not None else "ask"
    ledger = usage
    if ledger is None and session is not None:
        ledger = UsageLedger.from_dict(session.usage)

    lines = [
        f"workspace: {cwd}",
        f"session: {session.id if session is not None else '(none)'}",
        *model_lines(model),
        f"you spoke: {user_turns} time(s)",
        f"conversation: {used} / {limit} characters",
        f"memory notes: {notes}",
        f"remembered approvals: {writes} write path(s), {commands} command(s)",
        f"permission mode: {mode}",
        f"can undo this turn: {undoable} file(s)",
        f"restore points: {restore_points}",
        f"can redo: {redoable} step(s)",
    ]
    if ledger is not None and ledger.calls:
        lines.append(ledger.status_line())
    return "\n".join(lines)
