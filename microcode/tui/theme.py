from __future__ import annotations

from dataclasses import dataclass


def _rgb(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def _rgb_bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


@dataclass(frozen=True)
class ColorTheme:
    header: str
    session: str
    input: str
    approval: str
    user: str
    assistant: str
    progress: str
    tool: str
    tool_error: str
    command_highlight_bg: str
    reset: str = "\x1b[0m"
    bold: str = "\x1b[1m"
    dim: str = "\x1b[2m"
    italic: str = "\x1b[3m"
    reverse: str = "\x1b[7m"
    subtle: str = "\x1b[38;5;243m"
    highlight_bg: str = "\x1b[48;5;236m"


def _default_theme() -> ColorTheme:
    return ColorTheme(
        header=_rgb(120, 150, 140),
        session=_rgb(140, 120, 160),
        input=_rgb(130, 160, 100),
        approval=_rgb(170, 110, 110),
        user=_rgb(160, 130, 100),
        assistant=_rgb(100, 150, 150),
        progress=_rgb(170, 150, 90),
        tool=_rgb(140, 100, 160),
        tool_error=_rgb(180, 100, 100),
        command_highlight_bg=_rgb_bg(100, 110, 140),
    )


_THEME: ColorTheme | None = None


def theme() -> ColorTheme:
    global _THEME
    if _THEME is None:
        _THEME = _default_theme()
    return _THEME
