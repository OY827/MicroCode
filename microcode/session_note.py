from __future__ import annotations

import os

from microcode.checkpoint import load_groups
from microcode.session import Session
from microcode.types import ChatMessage

LAST_ASK_CHARS = 80
MAX_FILES = 8
SESSION_END_TAG = "session-end"


def session_note_disabled() -> bool:
    raw = (os.environ.get("MICROCODE_SESSION_NOTE") or "").strip().lower()
    return raw in {"0", "false", "no", "off"}


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _last_ask(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if text:
            return _clip(text, LAST_ASK_CHARS)
    return ""


def _changed_files(session: Session | None) -> list[str]:
    if session is None:
        return []
    seen: list[str] = []

    def _take(path: str) -> None:
        cleaned = path.strip().replace("\\", "/")
        if not cleaned or cleaned in seen:
            return
        if cleaned == ".microcode" or cleaned.startswith(".microcode/"):
            return
        seen.append(cleaned)

    for group in load_groups(session):
        for entry in group.entries:
            _take(entry.path)
    for item in session.checkpoint or []:
        if isinstance(item, dict):
            _take(str(item.get("path") or ""))
    return seen


def build_session_note(
    messages: list[ChatMessage] | None = None,
    session: Session | None = None,
) -> str:
    """Local one-liner: last ask + files touched. Empty when nothing happened."""

    ask = _last_ask(messages or [])
    files = _changed_files(session)
    if not ask and not files:
        return ""
    shown = files[:MAX_FILES]
    extra = len(files) - len(shown)
    file_text = ", ".join(shown)
    if extra > 0:
        file_text += f", and {extra} more"
    if ask and file_text:
        body = f'asked "{ask}"; changed {file_text}'
    elif ask:
        body = f'asked "{ask}"'
    else:
        body = f"changed {file_text}"
    if session is not None:
        return f"Last chat ({session.id}): {body}."
    return f"Last chat: {body}."


def finish_session_memory(memory, *, messages: list[ChatMessage], session: Session | None) -> str:
    """Write or refresh this chat's note, then apply end-of-chat tier rules."""

    lines: list[str] = []
    if not session_note_disabled():
        text = build_session_note(messages, session)
        if text:
            session_id = session.id if session is not None else ""
            if session_id:
                entry, created = memory.manager.upsert_tagged(
                    text,
                    tag=f"session:{session_id}",
                    extra_tags=[SESSION_END_TAG],
                )
                verb = "Remembered" if created else "Updated"
                lines.append(f"{verb} session note: - {entry.content}")
            else:
                memory.manager.add(text, tags=[SESSION_END_TAG])
                lines.append(f"Remembered session note: - {text}")
    lines.append(memory.maintain(session_end=True))
    return "\n".join(lines)
