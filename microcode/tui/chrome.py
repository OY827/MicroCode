from __future__ import annotations

import os
import re
import time
from functools import lru_cache

from microcode.tui.commands import SlashCommand
from microcode.tui.state import PendingPrompt
from microcode.tui.theme import theme

RESET = "\x1b[0m"
HIGHLIGHT_BG = "\x1b[48;5;236m"
BRIGHT_GREEN = "\x1b[92m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_WIDE_CHAR_PATTERN = re.compile(
    r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF"
    r"\uF900-\uFAFF\uFE10-\uFE19\uFE30-\uFE6F\uFF00-\uFF60\uFFE0-\uFFE6"
    r"\U0001F300-\U0001FAF6\U00020000-\U0003FFFD]"
)

_ts_cache: tuple[int, int] | None = None
_ts_cache_time: float = 0.0


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def invalidate_terminal_size_cache() -> None:
    global _ts_cache
    _ts_cache = None


def terminal_size() -> tuple[int, int]:
    global _ts_cache, _ts_cache_time
    now = time.monotonic()
    if _ts_cache is None or (now - _ts_cache_time) > 0.5:
        try:
            size = os.get_terminal_size()
            cols, rows = size.columns, size.lines
            _ts_cache = (100, 40) if cols <= 0 or rows <= 0 else (cols, rows)
        except (AttributeError, ValueError, OSError):
            _ts_cache = (100, 40)
        _ts_cache_time = now
    return _ts_cache


def char_display_width(char: str) -> int:
    if not char:
        return 0
    code = ord(char)
    if (
        0x1100 <= code <= 0x115F
        or code in {0x2329, 0x232A}
        or (0x2E80 <= code <= 0xA4CF and code != 0x303F)
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE10 <= code <= 0xFE19
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
        or 0x1F300 <= code <= 0x1FAF6
        or 0x20000 <= code <= 0x3FFFD
    ):
        return 2
    return 1


@lru_cache(maxsize=2048)
def _stripped_display_width(stripped: str) -> int:
    return len(stripped) + len(_WIDE_CHAR_PATTERN.findall(stripped))


def string_display_width(text: str) -> int:
    return _stripped_display_width(_ANSI_RE.sub("", text))


def truncate_plain(text: str, width: int) -> str:
    if string_display_width(text) <= width:
        return text
    limit = max(0, width - 3)
    result = ""
    used = 0
    index = 0
    while index < len(text):
        match = _ANSI_RE.match(text, index)
        if match:
            result += match.group()
            index = match.end()
            continue
        char = text[index]
        wide = char_display_width(char)
        if used + wide > limit:
            result += "..."
            index += 1
            while index < len(text):
                extra = _ANSI_RE.match(text, index)
                if extra:
                    result += extra.group()
                    index = extra.end()
                else:
                    index += 1
            return result
        result += char
        used += wide
        index += 1
    return result


def pad_plain(text: str, width: int) -> str:
    return text + (" " * max(0, width - string_display_width(text)))


def wrap_line(line: str, width: int) -> list[str]:
    inner = max(1, width)
    if string_display_width(line) <= inner:
        return [line]
    parts: list[str] = []
    current = ""
    current_width = 0
    for char in strip_ansi(line):
        wide = char_display_width(char)
        if current_width + wide > inner and current:
            parts.append(current)
            current = char
            current_width = wide
            continue
        current += char
        current_width += wide
    if current:
        parts.append(current)
    return parts or [""]


def border_line(kind: str, width: int, color: str) -> str:
    bar = "─" * max(0, width - 2)
    if kind == "top":
        return f"{color}╭{bar}╮{RESET}"
    if kind == "bottom":
        return f"{color}╰{bar}╯{RESET}"
    return f"{color}├{bar}┤{RESET}"


def panel_row(left: str, width: int, color: str, right: str | None = None) -> str:
    inner = max(1, width - 4)
    if right:
        left = truncate_plain(left, max(1, inner - string_display_width(right) - 1))
        gap = max(1, inner - string_display_width(left) - string_display_width(right))
        return f"{color}│{RESET} {left}{' ' * gap}{right} {color}│{RESET}"
    return f"{color}│{RESET} {pad_plain(truncate_plain(left, inner), inner)} {color}│{RESET}"


def render_banner(
    *,
    model: str,
    cwd: str,
    session_id: str,
    mode: str,
    message_count: int,
    width: int | None = None,
) -> str:
    colors = theme()
    cols = width or terminal_size()[0]
    cols = max(40, cols)
    title = f"{colors.bold}MicroCode{RESET}"
    right = f"{colors.subtle}full-screen{RESET}"
    line1 = f"model {model}   cwd {cwd}"
    line2 = f"session {session_id}   mode {mode}   msgs {message_count}"
    return "\n".join(
        [
            border_line("top", cols, colors.header),
            panel_row(title, cols, colors.header, right),
            panel_row(line1, cols, colors.header),
            panel_row(line2, cols, colors.header),
            border_line("bottom", cols, colors.header),
        ]
    )


def render_footer(status: str | None, width: int | None = None) -> str:
    colors = theme()
    cols = width or terminal_size()[0]
    left = status or "ready"
    right = "Enter send  Esc clear  ↑ history  PgUp scroll  Ctrl+C leave"
    return panel_row(
        f"{colors.progress}{left}{RESET}",
        max(40, cols),
        colors.header,
        f"{colors.subtle}{right}{RESET}",
    )


def render_input_prompt(current_input: str, cursor_offset: int, width: int | None = None) -> str:
    colors = theme()
    cols = width or terminal_size()[0]
    offset = max(0, min(cursor_offset, len(current_input)))
    prefix = f"{colors.input}{colors.bold}microcode>{RESET} "
    before = current_input[:offset]
    current = current_input[offset] if offset < len(current_input) else " "
    after = current_input[offset + 1 :]
    placeholder = "" if current_input else f"{colors.italic} Type a message or /help{RESET}"
    line = (
        f" {prefix}{before}{HIGHLIGHT_BG}{BRIGHT_GREEN}{current}{RESET}{after}"
        f"{colors.dim}{placeholder}{RESET}"
    )
    return panel_row(line, max(40, cols), colors.input)


def render_slash_menu(commands: list[SlashCommand], selected: int, width: int | None = None) -> str:
    colors = theme()
    cols = width or terminal_size()[0]
    cols = max(40, cols)
    lines = [border_line("top", cols, colors.input)]
    visible = commands[:8]
    for index, command in enumerate(visible):
        marker = "▸" if index == selected else " "
        label = f"{marker} {command.name}  {colors.subtle}{command.hint}{RESET}"
        if index == selected:
            label = f"{colors.command_highlight_bg}{label}{RESET}"
        lines.append(panel_row(label, cols, colors.input))
    lines.append(border_line("bottom", cols, colors.input))
    return "\n".join(lines)


def render_permission_prompt(pending: PendingPrompt, width: int | None = None) -> str:
    colors = theme()
    cols = width or terminal_size()[0]
    cols = max(40, cols)
    if pending.kind == "ask":
        body = [
            border_line("top", cols, colors.approval),
            panel_row(f"{colors.bold}Question{RESET}", cols, colors.approval),
            panel_row(pending.summary, cols, colors.approval),
            panel_row(f"answer> {pending.answer}_", cols, colors.approval),
            panel_row(f"{colors.subtle}Enter submit   Esc skip{RESET}", cols, colors.approval),
            border_line("bottom", cols, colors.approval),
        ]
        return "\n".join(body)
    if pending.kind == "edit":
        shown = pending.answer[-200:] if len(pending.answer) > 200 else pending.answer
        body = [
            border_line("top", cols, colors.approval),
            panel_row(f"{colors.bold}Edit file before write{RESET}", cols, colors.approval),
            panel_row(shown or "(empty file)", cols, colors.approval),
            panel_row(f"{colors.subtle}Enter writes this text   Esc back{RESET}", cols, colors.approval),
            border_line("bottom", cols, colors.approval),
        ]
        return "\n".join(body)
    body = [
        border_line("top", cols, colors.approval),
        panel_row(f"{colors.bold}Approve write or command{RESET}", cols, colors.approval),
        panel_row(pending.summary, cols, colors.approval),
        panel_row("[y] yes once   [a] always   [n] no   [e] edit   Esc deny", cols, colors.approval),
        border_line("bottom", cols, colors.approval),
    ]
    return "\n".join(body)
