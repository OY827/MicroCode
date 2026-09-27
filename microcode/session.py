from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from microcode.compact import context_chars
from microcode.types import ChatMessage

REPLAY_ITEM_LIMIT = 40
REPLAY_CLIP = 400
REPLAY_TOOL_CLIP = 200

LATEST_NAME = "latest"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class Session:
    id: str
    cwd: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage] = field(default_factory=list)
    checkpoint: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    undone: list[dict] = field(default_factory=list)
    model_name: str = ""
    usage: dict = field(default_factory=dict)
    permission_mode: str = "ask"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": list(self.messages),
            "checkpoint": list(self.checkpoint),
            "checkpoints": list(self.checkpoints),
            "undone": list(self.undone),
            "model_name": self.model_name,
            "usage": dict(self.usage),
            "permission_mode": self.permission_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        from microcode.permissions import normalize_permission_mode

        raw_checkpoint = data.get("checkpoint") or []
        raw_checkpoints = data.get("checkpoints") or []
        raw_undone = data.get("undone") or []
        raw_usage = data.get("usage") or {}
        return cls(
            id=str(data["id"]),
            cwd=str(data.get("cwd", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            messages=list(data.get("messages") or []),
            checkpoint=list(raw_checkpoint) if isinstance(raw_checkpoint, list) else [],
            checkpoints=list(raw_checkpoints) if isinstance(raw_checkpoints, list) else [],
            undone=list(raw_undone) if isinstance(raw_undone, list) else [],
            model_name=str(data.get("model_name") or ""),
            usage=dict(raw_usage) if isinstance(raw_usage, dict) else {},
            permission_mode=normalize_permission_mode(data.get("permission_mode")),
        )


class SessionStore:
    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace) / ".microcode" / "sessions"

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def create(self, cwd: str, messages: list[ChatMessage] | None = None) -> Session:
        stamp = _now()
        return Session(
            id=_new_id(),
            cwd=cwd,
            created_at=stamp,
            updated_at=stamp,
            messages=list(messages or []),
        )

    def save(self, session: Session) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        session.updated_at = _now()
        target = self.path_for(session.id)
        target.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.root / LATEST_NAME).write_text(session.id + "\n", encoding="utf-8")
        return target

    def load(self, session_id: str | None = None) -> Session:
        resolved = (session_id or "").strip() or self.latest_id()
        if not resolved:
            raise FileNotFoundError("No saved session found.")
        target = self.path_for(resolved)
        if not target.is_file():
            raise FileNotFoundError(f"Session not found: {resolved}")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid session file: {target}")
        return Session.from_dict(data)

    def latest_id(self) -> str | None:
        pointer = self.root / LATEST_NAME
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            if value and self.path_for(value).is_file():
                return value
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        return files[0].stem if files else None

    def list_sessions(self, limit: int = 10) -> list[Session]:
        if not self.root.is_dir():
            return []
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        sessions: list[Session] = []
        for path in files[:limit]:
            try:
                sessions.append(self.load(path.stem))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sessions


def summarize_session(session: Session) -> str:
    user_turns = sum(1 for message in session.messages if message.get("role") == "user")
    last_user = next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(session.messages)
            if message.get("role") == "user"
        ),
        "",
    )
    preview = last_user.replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    lines = [
        f"id: {session.id}",
        f"cwd: {session.cwd}",
        f"updated: {session.updated_at}",
        f"messages: {len(session.messages)}",
        f"chars: {context_chars(session.messages)}",
        f"user_turns: {user_turns}",
        f"undoable: {len(session.checkpoint)}",
        f"checkpoints: {len(session.checkpoints)}",
    ]
    if preview:
        lines.append(f"last_user: {preview}")
    return "\n".join(lines)


def summarize_session_list(sessions: list[Session]) -> str:
    if not sessions:
        return "No saved sessions."
    lines = []
    for session in sessions:
        user_turns = sum(1 for message in session.messages if message.get("role") == "user")
        lines.append(
            f"{session.id}  turns={user_turns}  messages={len(session.messages)}  {session.updated_at}"
        )
    return "\n".join(lines)


def _clip(text: object, limit: int) -> str:
    value = str(text or "").strip() or "(empty)"
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def format_replay(
    *,
    messages: list[ChatMessage] | None = None,
    session: Session | None = None,
    limit: int = REPLAY_ITEM_LIMIT,
) -> str:
    """Plain transcript of this chat. Skips the long system prompt."""

    msgs = list(messages if messages is not None else (session.messages if session is not None else []))
    visible = [message for message in msgs if message.get("role") != "system"]
    if not visible:
        header = f"replay: {session.id}" if session is not None else "replay"
        return f"{header}\nNo conversation to replay yet.\n"

    lines: list[str] = []
    if session is not None:
        lines.extend(
            [
                f"replay: {session.id}",
                f"workspace: {session.cwd}",
                f"created: {session.created_at}",
                f"updated: {session.updated_at}",
            ]
        )
        if session.model_name:
            lines.append(f"model: {session.model_name}")
    else:
        lines.append("replay: (unsaved)")
    lines.append("")

    omitted = 0
    if len(visible) > limit:
        omitted = len(visible) - limit
        visible = visible[-limit:]
        lines.append(f"... {omitted} earlier item(s) omitted")
        lines.append("")

    for index, message in enumerate(visible, start=omitted + 1):
        role = message.get("role")
        if role == "user":
            lines.append(f"[{index}] you")
            lines.append(_clip(message.get("content"), REPLAY_CLIP))
        elif role == "assistant":
            lines.append(f"[{index}] assistant")
            lines.append(_clip(message.get("content"), REPLAY_CLIP))
        elif role == "assistant_tool_call":
            lines.append(f"[{index}] tool {message.get('toolName') or '(unknown)'}")
            lines.append(_clip(json.dumps(message.get("input") or {}, ensure_ascii=False), REPLAY_TOOL_CLIP))
        elif role == "tool_result":
            flag = "error" if message.get("isError") else "ok"
            lines.append(f"[{index}] result [{flag}]")
            lines.append(_clip(message.get("content"), REPLAY_TOOL_CLIP))
        else:
            lines.append(f"[{index}] {role}")
            lines.append(_clip(message.get("content"), REPLAY_CLIP))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
