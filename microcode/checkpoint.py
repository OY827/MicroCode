from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from microcode.workspace import relative_workspace_path, workspace_root

PROTECTED_NAMES = {".env"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CheckpointEntry:
    path: str
    existed: bool
    content: str = ""
    is_dir: bool = False

    def to_dict(self) -> dict:
        data = {"path": self.path, "existed": self.existed, "content": self.content}
        if self.is_dir:
            data["is_dir"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointEntry:
        return cls(
            path=str(data.get("path", "")),
            existed=bool(data.get("existed", False)),
            content=str(data.get("content", "")),
            is_dir=bool(data.get("is_dir", False)),
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

    def record(self, path: str, *, existed: bool, content: str = "", is_dir: bool = False) -> None:
        if not path or path in self._by_path:
            return
        self._by_path[path] = CheckpointEntry(
            path=path,
            existed=existed,
            content=content if existed else "",
            is_dir=is_dir,
        )

    def record_target(
        self,
        cwd: str,
        target: Path,
        *,
        existed: bool,
        content: str = "",
        is_dir: bool = False,
    ) -> None:
        self.record(
            relative_workspace_path(cwd, target),
            existed=existed,
            content=content,
            is_dir=is_dir,
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

    def entries(self) -> list[CheckpointEntry]:
        return list(self._by_path.values())

    def restore(self, cwd: str) -> list[str]:
        root = workspace_root(cwd)
        entries = list(self._by_path.values())
        # Recreate old paths before deleting paths this turn created (needed for move).
        ordered = [entry for entry in entries if entry.existed] + [
            entry for entry in entries if not entry.existed
        ]
        return [_restore_entry(root, entry) for entry in ordered]


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
        if entry.is_dir:
            try:
                normalized.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                return f"failed {entry.path}: {error}"
            return f"restored {entry.path}"
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
    try:
        if entry.is_dir:
            if not normalized.is_dir():
                return f"skipped {entry.path} (not a directory)"
            shutil.rmtree(normalized)
            return f"deleted {entry.path}"
        if normalized.is_dir():
            return f"skipped {entry.path} (path is a directory)"
        normalized.unlink()
    except OSError as error:
        return f"failed {entry.path}: {error}"
    return f"deleted {entry.path}"


@dataclass
class CheckpointGroup:
    """One user turn's pre-write snapshots, kept so earlier turns can be rewound."""

    id: str
    created_at: str
    entries: list[CheckpointEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointGroup:
        raw_entries = data.get("entries") or []
        entries = [
            CheckpointEntry.from_dict(item)
            for item in raw_entries
            if isinstance(item, dict)
        ]
        return cls(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            entries=entries,
        )


def load_groups(session) -> list[CheckpointGroup]:
    raw = getattr(session, "checkpoints", None) or []
    groups: list[CheckpointGroup] = []
    if not isinstance(raw, list):
        return groups
    for item in raw:
        if isinstance(item, dict) and item.get("id"):
            groups.append(CheckpointGroup.from_dict(item))
    return groups


def _next_group_id(groups: list[CheckpointGroup]) -> str:
    highest = 0
    for group in groups:
        if group.id.startswith("cp_"):
            try:
                highest = max(highest, int(group.id[3:]))
            except ValueError:
                continue
    return f"cp_{highest + 1}"


def remember_write_group(session, checkpoint: WriteCheckpoint) -> CheckpointGroup | None:
    """Append this turn's snapshots to the rewind history."""

    if not checkpoint:
        return None
    session.undone = []
    groups = load_groups(session)
    group = CheckpointGroup(
        id=_next_group_id(groups),
        created_at=_now(),
        entries=checkpoint.entries(),
    )
    groups.append(group)
    session.checkpoints = [item.to_dict() for item in groups]
    session.checkpoint = checkpoint.to_list()
    return group


def snapshot_paths(cwd: str, entries: list[CheckpointEntry]) -> list[CheckpointEntry]:
    """Capture current file state for the given paths so a rewind can be redone."""

    root = workspace_root(cwd)
    snapped: list[CheckpointEntry] = []
    seen: set[str] = set()
    for entry in entries:
        path = entry.path
        if not path or path in seen:
            continue
        seen.add(path)
        snapped.append(_snapshot_one(root, path))
    return snapped


def _snapshot_one(root: Path, path: str) -> CheckpointEntry:
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        normalized = target.resolve()
        normalized.relative_to(root)
    except (OSError, ValueError):
        return CheckpointEntry(path=path, existed=False)
    if not normalized.exists():
        return CheckpointEntry(path=path, existed=False)
    if normalized.is_dir():
        return CheckpointEntry(path=path, existed=True, is_dir=True)
    try:
        content = normalized.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        content = ""
    return CheckpointEntry(path=path, existed=True, content=content)


def _load_undone(session) -> list[dict]:
    raw = getattr(session, "undone", None) or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _push_undone(
    session,
    *,
    groups: list[CheckpointGroup],
    after: list[CheckpointEntry],
    checkpoint: list[dict],
) -> None:
    undone = _load_undone(session)
    undone.append(
        {
            "groups": [group.to_dict() for group in groups],
            "after": [entry.to_dict() for entry in after],
            "checkpoint": list(checkpoint),
        }
    )
    session.undone = undone


def format_checkpoints(session) -> str:
    groups = load_groups(session)
    undone = _load_undone(session)
    if not groups:
        if undone:
            return f"No checkpoints.\ncan redo: {len(undone)} step(s)"
        return "No checkpoints."
    lines = []
    for group in groups:
        files = ", ".join(entry.path for entry in group.entries) or "(no files)"
        lines.append(f"{group.id}  {group.created_at}  {files}")
    if undone:
        lines.append(f"can redo: {len(undone)} step(s)")
    return "\n".join(lines)


def parse_rewind_target(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    target = parts[1].strip()
    return target or None


def _select_groups(groups: list[CheckpointGroup], target_id: str | None) -> list[CheckpointGroup] | None:
    if not groups:
        return []
    if target_id is None:
        return [groups[-1]]
    needle = target_id.strip().lower()
    for index, group in enumerate(groups):
        if group.id.lower() == needle:
            return groups[index:]
    return None


def _entries_before_selected(selected: list[CheckpointGroup]) -> list[CheckpointEntry]:
    """For each path, keep the oldest selected snapshot — the state before that first write."""

    by_path: dict[str, CheckpointEntry] = {}
    for group in selected:
        for entry in group.entries:
            if entry.path and entry.path not in by_path:
                by_path[entry.path] = entry
    return list(by_path.values())


def preview_rewind(*, session, target_id: str | None = None) -> str:
    selected = _select_groups(load_groups(session), target_id)
    if selected is None:
        return f"Checkpoint not found: {target_id}"
    if not selected:
        return "Nothing to rewind."
    entries = _entries_before_selected(selected)
    lines = [
        f"Would restore {len(selected)} checkpoint(s), {len(entries)} file(s).",
        f"From {selected[0].id} through {selected[-1].id}.",
    ]
    for entry in entries:
        action = f"restore {entry.path}" if entry.existed else f"delete {entry.path}"
        lines.append(f"  {action}")
    return "\n".join(lines)


def rewind_writes(*, session, store=None, cwd: str, target_id: str | None = None) -> str:
    """Restore files to the state before the chosen checkpoint (and every write after it)."""

    groups = load_groups(session)
    selected = _select_groups(groups, target_id)
    if selected is None:
        return f"Checkpoint not found: {target_id}"
    if not selected:
        return "Nothing to rewind."
    entries = _entries_before_selected(selected)
    after = snapshot_paths(cwd, entries)
    previous_checkpoint = list(getattr(session, "checkpoint", None) or [])
    _push_undone(session, groups=selected, after=after, checkpoint=previous_checkpoint)
    lines = WriteCheckpoint(entries).restore(cwd)
    keep = groups[: len(groups) - len(selected)]
    session.checkpoints = [group.to_dict() for group in keep]
    session.checkpoint = []
    if store is not None:
        store.save(session)
    return "Rewind:\n" + "\n".join(lines)


def undo_last_writes(*, session, store=None, cwd: str) -> str:
    """Restore the last write group. Leaves messages unchanged."""

    checkpoint = WriteCheckpoint.from_list(getattr(session, "checkpoint", None))
    if not checkpoint:
        return "Nothing to undo."
    after = snapshot_paths(cwd, checkpoint.entries())
    previous_checkpoint = list(getattr(session, "checkpoint", None) or [])
    groups = load_groups(session)
    popped: list[CheckpointGroup] = []
    if groups:
        popped = [groups.pop()]
        session.checkpoints = [group.to_dict() for group in groups]
    _push_undone(session, groups=popped, after=after, checkpoint=previous_checkpoint)
    lines = checkpoint.restore(cwd)
    session.checkpoint = []
    if store is not None:
        store.save(session)
    return "Undo:\n" + "\n".join(lines)


def redo_writes(*, session, store=None, cwd: str) -> str:
    """Re-apply the last undone or rewound write group."""

    undone = _load_undone(session)
    if not undone:
        return "Nothing to redo."
    record = undone.pop()
    session.undone = undone
    after_raw = record.get("after") or []
    after = [
        CheckpointEntry.from_dict(item)
        for item in after_raw
        if isinstance(item, dict)
    ]
    lines = WriteCheckpoint(after).restore(cwd) if after else []
    groups = load_groups(session)
    restored = [
        CheckpointGroup.from_dict(item)
        for item in record.get("groups") or []
        if isinstance(item, dict) and item.get("id")
    ]
    groups.extend(restored)
    session.checkpoints = [group.to_dict() for group in groups]
    previous = record.get("checkpoint")
    session.checkpoint = list(previous) if isinstance(previous, list) else []
    if store is not None:
        store.save(session)
    if not lines:
        return "Redo:\nnothing to restore"
    return "Redo:\n" + "\n".join(lines)
