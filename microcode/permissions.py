from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["write", "command"]


@dataclass
class PermissionStore:
    """Remembered allow rules for writes and commands in this workspace."""

    path: Path
    writes: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    def is_allowed(self, kind: Kind, key: str) -> bool:
        normalized = _normalize_key(kind, key)
        if not normalized:
            return False
        bucket = self._bucket(kind)
        return any(_keys_match(kind, item, normalized) for item in bucket)

    def allow(self, kind: Kind, key: str) -> bool:
        normalized = _normalize_key(kind, key)
        if not normalized or self.is_allowed(kind, normalized):
            return False
        self._bucket(kind).append(normalized)
        return True

    def remove(self, kind: Kind, key: str) -> bool:
        normalized = _normalize_key(kind, key)
        bucket = self._bucket(kind)
        before = len(bucket)
        kept = [item for item in bucket if not _keys_match(kind, item, normalized)]
        if kind == "write":
            self.writes = kept
        else:
            self.commands = kept
        return len(kept) < before

    def clear(self) -> None:
        self.writes = []
        self.commands = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"writes": list(self.writes), "commands": list(self.commands)}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def format(self) -> str:
        if not self.writes and not self.commands:
            return (
                f"{self.path}\n"
                "No remembered permissions.\n"
                "Approve with 'a' to always allow a write or command."
            )
        lines = [str(self.path)]
        if self.writes:
            lines.append("writes:")
            lines.extend(f"  {item}" for item in self.writes)
        if self.commands:
            lines.append("commands:")
            lines.extend(f"  {item}" for item in self.commands)
        return "\n".join(lines)

    def _bucket(self, kind: Kind) -> list[str]:
        return self.writes if kind == "write" else self.commands

    @classmethod
    def load(cls, cwd: str | Path) -> PermissionStore:
        path = Path(cwd) / ".microcode" / "permissions.json"
        if not path.is_file():
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        if not isinstance(data, dict):
            return cls(path=path)
        writes = [str(item) for item in data.get("writes") or [] if str(item).strip()]
        commands = [str(item) for item in data.get("commands") or [] if str(item).strip()]
        return cls(path=path, writes=writes, commands=commands)


def parse_decision(decision: Any) -> tuple[bool, bool]:
    """Return (allowed, remember). Bool callbacks stay once-only."""

    if decision is True:
        return True, False
    if decision is False or decision is None:
        return False, False
    text = str(decision).strip().lower()
    if text in {"a", "always"}:
        return True, True
    if text in {"y", "yes", "allow"}:
        return True, False
    return False, False


def parse_permissions_command(text: str) -> tuple[str, str, str]:
    """Return (action, kind, key) for a /permissions line."""

    rest = text.strip()[len("/permissions") :].strip()
    if not rest:
        return "show", "", ""
    lowered = rest.lower()
    if lowered == "clear":
        return "clear", "", ""
    if lowered.startswith("remove "):
        payload = rest[7:].strip()
        kind, _, key = payload.partition(" ")
        kind = kind.strip().lower()
        key = key.strip()
        if kind in {"write", "command"} and key:
            return "remove", kind, key
        return "help", "", ""
    return "help", "", ""


def _normalize_key(kind: Kind, key: str) -> str:
    text = key.strip()
    if kind == "write":
        return text.replace("\\", "/")
    return text


def _keys_match(kind: Kind, stored: str, candidate: str) -> bool:
    if kind == "write":
        return stored.replace("\\", "/") == candidate.replace("\\", "/")
    return stored == candidate
