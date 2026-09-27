from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from microcode.types import ChatMessage

DEFAULT_MAX_CHARS = 80_000
KEEP_RECENT_TURNS = 1
PREVIEW_CHARS = 400
HEAD_CHARS = 200
TAIL_CHARS = 200
COMPACT_PREFIX = "[compacted]"
RESULTS_DIR = Path(".microcode") / "tool-results"
KEEP_RESULT_DAYS = 7
KEEP_RESULT_FILES = 40
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
COMPACT_HINT = (
    "If you see a tool result starting with [compacted] that includes "
    "'saved .microcode/tool-results/...', call read_file on that saved path "
    "to recover the full text. The stub is only a pointer, not the original output."
)


@dataclass(frozen=True)
class CompactReport:
    before: int
    after: int
    compacted: int
    spilled: int = 0
    cleaned: int = 0
    forced: bool = False

    def summary(self) -> str:
        if self.compacted == 0 and self.cleaned == 0:
            return "Nothing to compact."
        parts = []
        if self.compacted:
            extra = f", saved {self.spilled} tool result(s) " if self.spilled else " "
            parts.append(
                f"context: compacted {self.compacted} messages{extra}"
                f"({self.before} → {self.after} chars)"
            )
        if self.cleaned:
            parts.append(f"removed {self.cleaned} old tool-result file(s)")
        return ". ".join(parts)


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
    cwd: str | None = None,
) -> tuple[list[ChatMessage], CompactReport]:
    """Shrink older tool payloads. System and the latest user turn stay intact.

    Fat tool results are written under .microcode/tool-results/ when cwd is set.
    Previews keep a head and a tail so errors at the end are not dropped.
    """

    current = [dict(message) for message in messages]
    before = context_chars(current)
    cleaned = cleanup_tool_results(cwd) if cwd else 0
    if not force and (max_chars <= 0 or before <= max_chars):
        return current, CompactReport(before=before, after=before, compacted=0, cleaned=cleaned)

    protect_from = _recent_start_index(current, keep_recent_turns)
    compacted = 0
    spilled = 0
    for index, message in enumerate(current):
        if index >= protect_from:
            break
        next_message, wrote = _compact_one(message, cwd=cwd)
        if next_message is not message:
            current[index] = next_message
            compacted += 1
            if wrote:
                spilled += 1

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
        spilled=spilled,
        cleaned=cleaned,
        forced=force,
    )


def cleanup_tool_results(
    cwd: str | None,
    *,
    keep_days: int = KEEP_RESULT_DAYS,
    keep_latest: int = KEEP_RESULT_FILES,
) -> int:
    """Delete spilled tool results older than keep_days, or beyond the newest keep_latest."""

    if not cwd:
        return 0
    folder = Path(cwd) / RESULTS_DIR
    if not folder.is_dir():
        return 0
    try:
        files = [path for path in folder.iterdir() if path.is_file()]
    except OSError:
        return 0
    if not files:
        return 0
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    cutoff = time.time() - max(keep_days, 0) * 86400
    removed = 0
    for index, path in enumerate(files):
        try:
            too_old = path.stat().st_mtime < cutoff
        except OSError:
            continue
        too_many = index >= max(keep_latest, 0)
        if not too_old and not too_many:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


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


def _compact_one(message: ChatMessage, *, cwd: str | None) -> tuple[ChatMessage, bool]:
    role = message.get("role")
    if role == "system" or _is_compacted(message):
        return message, False
    if role == "tool_result":
        if message_chars(message) <= PREVIEW_CHARS:
            return message, False
        stub, wrote = _summarize_tool_result(message, cwd=cwd)
        return {**message, "content": stub}, wrote
    if role == "assistant_tool_call":
        next_message = _compact_tool_call(message)
        return next_message, False
    return message, False


def _compact_text_message(message: ChatMessage) -> ChatMessage:
    role = message.get("role")
    if role not in {"user", "assistant"} or _is_compacted(message):
        return message
    content = str(message.get("content", ""))
    if len(content) <= HEAD_CHARS + TAIL_CHARS:
        return message
    preview = _preview_ends(content, head=HEAD_CHARS, tail=TAIL_CHARS)
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
            "preview": _preview_ends(raw),
        },
    }


def _summarize_tool_result(message: ChatMessage, *, cwd: str | None) -> tuple[str, bool]:
    content = str(message.get("content", ""))
    name = str(message.get("toolName") or "tool")
    flag = " error" if message.get("isError") else ""
    saved = _spill_text(content, name, cwd)
    preview = _preview_ends(content)
    prefix = f"{COMPACT_PREFIX} {name}{flag} ({len(content)} chars)"
    if saved:
        return f"{prefix}: saved {saved} | {preview}", True
    return f"{prefix}: {preview}", False


def _spill_text(content: str, tool_name: str, cwd: str | None) -> str | None:
    if not cwd:
        return None
    safe = _SAFE_NAME.sub("-", tool_name).strip("-") or "tool"
    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    relative = RESULTS_DIR / f"{stamp}-{safe}-{digest}.txt"
    target = Path(cwd) / relative
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError:
        return None
    return relative.as_posix()


def _is_compacted(message: ChatMessage) -> bool:
    if str(message.get("content", "")).startswith(COMPACT_PREFIX):
        return True
    payload = message.get("input")
    return isinstance(payload, dict) and bool(payload.get("_compacted"))


def _preview_ends(text: str, *, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= head + tail + 3:
        return collapsed
    return f"{collapsed[:head]} … {collapsed[-tail:]}"


def _dump(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
