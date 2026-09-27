from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from microcode.types import AgentStep

TAPE_NAME = "turns.jsonl"
CLIP = 180
DEFAULT_LIMIT = 20
MAX_LIMIT = 200
_LOCK = threading.Lock()


def tape_enabled() -> bool:
    """On unless MICROCODE_TURN_TAPE=0. Tests set that so pytest stays quiet."""

    return os.environ.get("MICROCODE_TURN_TAPE", "1").strip() != "0"


def tape_path(cwd: str | Path) -> Path:
    return Path(cwd) / ".microcode" / TAPE_NAME


def open_turn_tape(
    cwd: str | Path,
    session_id: str = "",
    *,
    via: str = "",
) -> TurnTape | None:
    if not tape_enabled():
        return None
    return TurnTape(cwd, session_id=session_id, via=via)


def clip_text(value: Any, limit: int = CLIP) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def preview_input(value: Any) -> str:
    if isinstance(value, dict):
        parts = [f"{key}={clip_text(item, 60)}" for key, item in list(value.items())[:6]]
        return clip_text(" ".join(parts))
    return clip_text(value)


class TurnTape:
    """Append-only log of one workspace's model decisions and tool results."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        session_id: str = "",
        via: str = "",
        path: Path | None = None,
    ) -> None:
        self.cwd = str(cwd)
        self.session_id = session_id
        self.via = via
        self.path = path if path is not None else tape_path(cwd)

    def spawn(self) -> TurnTape:
        """Same file, marked as a nested sub-agent."""

        via = self.via or "subagent"
        return TurnTape(self.cwd, session_id=self.session_id, via=via, path=self.path)

    def record_user(self, text: str) -> None:
        self._append({"kind": "user", "text": clip_text(text)})

    def record_model(self, *, model: str, step: AgentStep, roles: list[str]) -> None:
        calls = [
            f"{call['toolName']} {preview_input(call.get('input'))}".strip()
            for call in (step.calls or [])[:6]
        ]
        self._append(
            {
                "kind": "model",
                "model": model or "unknown",
                "decision": step.type,
                "content": clip_text(step.content) if step.type == "assistant" else "",
                "calls": calls,
                "roles": [str(role) for role in roles[-12:]],
            }
        )

    def record_tool(self, *, name: str, tool_input: Any, ok: bool, output: str) -> None:
        self._append(
            {
                "kind": "tool",
                "tool": name,
                "ok": bool(ok),
                "input": preview_input(tool_input),
                "output": clip_text(output),
            }
        )

    def record_stop(self, reason: str, *, detail: str = "", tool: str = "") -> None:
        payload: dict[str, Any] = {"kind": "stop", "reason": reason}
        if detail:
            payload["detail"] = clip_text(detail, 80)
        if tool:
            payload["tool"] = tool
        self._append(payload)

    def record_compact(self, summary: str) -> None:
        self._append({"kind": "compact", "summary": clip_text(summary)})

    def _append(self, payload: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session": self.session_id,
        }
        if self.via:
            record["via"] = self.via
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        try:
            with _LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError:
            return


def load_tape(cwd: str | Path) -> list[dict[str, Any]]:
    path = tape_path(cwd)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def format_record(index: int, record: dict[str, Any]) -> str:
    stamp = str(record.get("ts") or "")
    clock = stamp[11:19] if len(stamp) >= 19 else stamp
    via = f" via {record['via']}" if record.get("via") else ""
    kind = record.get("kind")
    if kind == "user":
        return f"[{index}] {clock} you{via}  {record.get('text', '')}"
    if kind == "model":
        model = record.get("model") or "model"
        if record.get("decision") == "tool_calls":
            calls = ", ".join(str(item) for item in (record.get("calls") or []))
            return f"[{index}] {clock} model {model}{via} -> {calls or 'tools'}"
        content = record.get("content") or "(empty)"
        return f"[{index}] {clock} model {model}{via} -> {content}"
    if kind == "tool":
        flag = "ok" if record.get("ok") else "error"
        tool_input = record.get("input") or ""
        output = record.get("output") or ""
        extra = "  ".join(part for part in (tool_input, output) if part)
        return f"[{index}] {clock} tool {record.get('tool', '?')} [{flag}]{via}  {extra}".rstrip()
    if kind == "compact":
        return f"[{index}] {clock} compact{via}  {record.get('summary', '')}"
    if kind == "stop":
        detail = record.get("detail") or record.get("tool") or ""
        suffix = f"  {detail}" if detail else ""
        return f"[{index}] {clock} stop {record.get('reason', '')}{via}{suffix}"
    return f"[{index}] {clock} {kind}{via}"


def format_tape(cwd: str | Path, *, limit: int = DEFAULT_LIMIT) -> str:
    path = tape_path(cwd)
    rows = load_tape(cwd)
    if not rows:
        return f"No turn tape yet.\n{path}\n"
    window = max(1, min(int(limit), MAX_LIMIT))
    start = max(0, len(rows) - window)
    lines = [
        f"turn tape: {path}",
        f"showing {start + 1}-{len(rows)} of {len(rows)}",
    ]
    for offset, record in enumerate(rows[start:], start=start + 1):
        lines.append(format_record(offset, record))
    return "\n".join(lines) + "\n"


def parse_tape_args(text: str) -> tuple[str, int]:
    """Return ('show', n), ('path', 0), or ('usage', 0)."""

    arg = text[len("/tape") :].strip().lower()
    if not arg:
        return "show", DEFAULT_LIMIT
    if arg == "path":
        return "path", 0
    if arg.isdigit():
        count = int(arg)
        if count < 1:
            return "usage", 0
        return "show", min(count, MAX_LIMIT)
    return "usage", 0
