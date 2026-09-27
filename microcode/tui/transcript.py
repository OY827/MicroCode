from __future__ import annotations

from microcode.tui.chrome import RESET, panel_row, string_display_width, terminal_size, wrap_line
from microcode.tui.theme import theme
from microcode.tui.types import TranscriptEntry

_TOOL_PREVIEW_LINES = 6


def _kind_prefix(entry: TranscriptEntry) -> str:
    colors = theme()
    if entry.kind == "user":
        return f"{colors.user}you{RESET}"
    if entry.kind == "tool":
        name = entry.tool_name or "tool"
        status = entry.status or "running"
        color = colors.tool_error if status == "error" else colors.tool
        mark = {"running": "…", "success": "ok", "error": "err"}.get(status, status)
        return f"{color}{name} [{mark}]{RESET}"
    if entry.kind == "progress":
        return f"{colors.progress}…{RESET}"
    return f"{colors.assistant}microcode{RESET}"


def entry_body_lines(entry: TranscriptEntry, width: int) -> list[str]:
    inner = max(20, width - 4)
    if entry.kind == "tool" and entry.collapsed:
        summary = entry.collapsed_summary or "output collapsed"
        return wrap_line(f"{_kind_prefix(entry)}  {summary}", inner)
    body = entry.body or ""
    if entry.kind == "tool" and not entry.collapsed:
        lines = body.splitlines() or [body]
        if len(lines) > _TOOL_PREVIEW_LINES:
            body = "\n".join(lines[:_TOOL_PREVIEW_LINES] + ["..."])
    prefix = _kind_prefix(entry)
    raw_lines = (body or "").splitlines() or [""]
    visual: list[str] = []
    for index, line in enumerate(raw_lines):
        lead = f"{prefix}  " if index == 0 else " " * (string_display_width(prefix) + 2)
        visual.extend(wrap_line(lead + line, inner))
    return visual or [prefix]


def transcript_visual_lines(entries: list[TranscriptEntry], width: int) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        if lines:
            lines.append("")
        lines.extend(entry_body_lines(entry, width))
    return lines


def max_scroll_offset(entries: list[TranscriptEntry], width: int, height: int) -> int:
    total = len(transcript_visual_lines(entries, width))
    return max(0, total - max(1, height))


def render_transcript(
    entries: list[TranscriptEntry],
    *,
    scroll_offset: int = 0,
    width: int | None = None,
    height: int | None = None,
) -> str:
    colors = theme()
    cols, rows = terminal_size()
    cols = width or cols
    rows = height or max(4, rows - 12)
    cols = max(40, cols)
    visual = transcript_visual_lines(entries, cols)
    if not visual:
        visual = [f"{colors.subtle}Waiting for the first message.{RESET}"]
    max_offset = max(0, len(visual) - rows)
    offset = max(0, min(scroll_offset, max_offset))
    start = max(0, len(visual) - rows - offset)
    window = visual[start : start + rows]
    while len(window) < rows:
        window.append("")
    return "\n".join(panel_row(line, cols, colors.session) for line in window)
