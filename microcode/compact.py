from __future__ import annotations

import json
import os
from dataclasses import dataclass

from microcode.types import ChatMessage

DEFAULT_MAX_CHARS = 80_000
KEEP_RECENT_TURNS = 1
PREVIEW_CHARS = 400
COMPACT_PREFIX = "[compacted]"


@dataclass(frozen=True)
class CompactReport:
    before: int
    after: int
    compacted: int
    forced: bool = False

    def summary(self) -> str:
        if self.compacted == 0:
            return "Nothing to compact."
        return (
            f"context: compacted {self.compacted} messages "
            f"({self.before} → {self.after} chars)"
        )


def compact_limit() -> int:
    raw = (os.environ.get("MICROCODE_COMPACT_CHARS") or "").strip()
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_MAX_CHARS


def message_chars(message: ChatMessage) -> int:
    total = len(str(message.get("content", "")))
    payload = message.get("input")
    if payload is not None:
        total += len(_dump(payload))
    return total


def context_chars(messages: list[ChatMessage]) -> int:
    return sum(message_chars(message) for message in messages)


def compact_messages(
    messages: list[ChatMessage],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    keep_recent_turns: int = KEEP_RECENT_TURNS,
    force: bool = False,
) -> tuple[list[ChatMessage], CompactReport]:
    """Shrink older tool payloads. System and the latest user turn stay intact."""

    current = [dict(message) for message in messages]
    before = context_chars(current)
    if not force and (max_chars <= 0 or before <= max_chars):
        return current, CompactReport(before=before, after=before, compacted=0)

    protect_from = _recent_start_index(current, keep_recent_turns)
    compacted = 0
    for index, message in enumerate(current):
        if index >= protect_from:
            break
        next_message = _compact_one(message)
        if next_message is not message:
            current[index] = next_message
            compacted += 1

    after = context_chars(current)
    if after > max_chars and max_chars > 0:
        for index, message in enumerate(current):
            if index >= protect_from:
                break
            next_message = _compact_text_message(message)
            if next_message is not message:
                current[index] = next_message
                compacted += 1
            after = context_chars(current)
            if after <= max_chars and not force:
                break

    return current, CompactReport(
        before=before,
        after=after,
        compacted=compacted,
        forced=force,
    )


def _recent_start_index(messages: list[ChatMessage], keep_recent_turns: int) -> int:
    if keep_recent_turns <= 0:
        return len(messages)
    seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            seen += 1
            if seen >= keep_recent_turns:
                return index
    return 0


def _compact_one(message: ChatMessage) -> ChatMessage:
    role = message.get("role")
    if role == "system" or _is_compacted(message):
        return message
    if role == "tool_result":
        if message_chars(message) > PREVIEW_CHARS:
            return {**message, "content": _summarize_tool_result(message)}
        return message
    if role == "assistant_tool_call":
        return _compact_tool_call(message)
    return message


def _compact_text_message(message: ChatMessage) -> ChatMessage:
    role = message.get("role")
    if role not in {"user", "assistant"} or _is_compacted(message):
        return message
    content = str(message.get("content", ""))
    if len(content) <= PREVIEW_CHARS:
        return message
    preview = _preview(content)
    return {
        **message,
        "content": f"{COMPACT_PREFIX} {role} ({len(content)} chars): {preview}",
    }


def _compact_tool_call(message: ChatMessage) -> ChatMessage:
    payload = message.get("input")
    raw = _dump(payload) if payload is not None else ""
    if len(raw) <= PREVIEW_CHARS:
        return message
    if not raw:
        return message
    path = ""
    if isinstance(payload, dict):
        if payload.get("_compacted"):
            return message
        path = str(payload.get("path") or "")
    return {
        **message,
        "input": {
            "_compacted": True,
            "path": path,
            "original_chars": len(raw),
            "preview": _preview(raw),
        },
    }


def _summarize_tool_result(message: ChatMessage) -> str:
    content = str(message.get("content", ""))
    name = str(message.get("toolName") or "tool")
    flag = " error" if message.get("isError") else ""
    return f"{COMPACT_PREFIX} {name}{flag} ({len(content)} chars): {_preview(content)}"


def _is_compacted(message: ChatMessage) -> bool:
    if str(message.get("content", "")).startswith(COMPACT_PREFIX):
        return True
    payload = message.get("input")
    return isinstance(payload, dict) and bool(payload.get("_compacted"))


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= PREVIEW_CHARS:
        return collapsed
    return collapsed[:PREVIEW_CHARS] + "..."


def _dump(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
