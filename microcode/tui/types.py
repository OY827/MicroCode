from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EntryKind = Literal["user", "assistant", "progress", "tool"]
EntryStatus = Literal["running", "success", "error"]


@dataclass(slots=True)
class TranscriptEntry:
    id: int
    kind: EntryKind
    body: str
    tool_name: str | None = None
    status: EntryStatus | None = None
    collapsed: bool = False
    collapsed_summary: str | None = None
