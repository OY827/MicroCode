from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from microcode.workspace import relative_workspace_path, workspace_root

PROTECTED_NAMES = {".env"}


@dataclass
class CheckpointEntry:
    path: str
    existed: bool
    content: str = ""

    def to_dict(self) -> dict:
        return {"path": self.path, "existed": self.existed, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointEntry:
        return cls(
            path=str(data.get("path", "")),
            existed=bool(data.get("existed", False)),
            content=str(data.get("content", "")),
        )


class WriteCheckpoint:
    """Pre-write snapshots for one user turn. The first snapshot per path wins."""

    def __init__(self, entries: list[CheckpointEntry] | None = None) -> None:
        self._by_path: dict[str, CheckpointEntry] = {}
        for entry in entries or []:
            if entry.path and entry.path not in self._by_path:
                self._by_path[entry.path] = entry

    def __bool__(self) -> bool:
        return bool(self._by_path)

    def record(self, path: str, *, existed: bool, content: str = "") -> None:
        if not path or path in self._by_path:
            return
        self._by_path[path] = CheckpointEntry(
            path=path,
            existed=existed,
            content=content if existed else "",
        )

    def record_target(self, cwd: str, target: Path, *, existed: bool, content: str = "") -> None:
        self.record(
            relative_workspace_path(cwd, target),
            existed=existed,
            content=content,
        )

    def to_list(self) -> list[dict]:
        return [entry.to_dict() for entry in self._by_path.values()]

    @classmethod
    def from_list(cls, data: list | None) -> WriteCheckpoint:
        if not data:
            return cls()
        entries = [
            CheckpointEntry.from_dict(item)
            for item in data
            if isinstance(item, dict)
        ]
        return cls(entries)

    def restore(self, cwd: str) -> list[str]:
        root = workspace_root(cwd)
        lines: list[str] = []
        for entry in self._by_path.values():
            lines.append(_restore_entry(root, entry))
        return lines


def _restore_entry(root: Path, entry: CheckpointEntry) -> str:
    if not entry.path or Path(entry.path).name in PROTECTED_NAMES:
        return f"skipped {entry.path or '(empty path)'} (protected or invalid)"

    candidate = Path(entry.path)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        normalized = target.resolve()
        normalized.relative_to(root)
    except (OSError, ValueError):
        return f"skipped {entry.path} (escapes workspace)"

    if normalized.name in PROTECTED_NAMES:
        return f"skipped {entry.path} (protected)"

    if entry.existed:
        if normalized.exists() and normalized.is_dir():
            return f"skipped {entry.path} (path is a directory)"
        try:
            normalized.parent.mkdir(parents=True, exist_ok=True)
            normalized.write_text(entry.content, encoding="utf-8")
        except OSError as error:
            return f"failed {entry.path}: {error}"
        return f"restored {entry.path}"

    if not normalized.exists():
        return f"already absent {entry.path}"
    if normalized.is_dir():
        return f"skipped {entry.path} (path is a directory)"
    try:
        normalized.unlink()
    except OSError as error:
        return f"failed {entry.path}: {error}"
    return f"deleted {entry.path}"


def undo_last_writes(*, session, store=None, cwd: str) -> str:
    """Restore the last write group. Leaves messages unchanged."""

    checkpoint = WriteCheckpoint.from_list(getattr(session, "checkpoint", None))
    if not checkpoint:
        return "Nothing to undo."
    lines = checkpoint.restore(cwd)
    session.checkpoint = []
    if store is not None:
        store.save(session)
    return "Undo:\n" + "\n".join(lines)
