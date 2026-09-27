from __future__ import annotations

import os
import sys

ENTER_ALT_SCREEN = "\u001b[?1049h"
EXIT_ALT_SCREEN = "\u001b[?1049l"
ERASE_SCREEN_AND_HOME = "\u001b[2J\u001b[H"
ENABLE_MOUSE_TRACKING = "\u001b[?1000h\u001b[?1003h\u001b[?1006h"
DISABLE_MOUSE_TRACKING = "\u001b[?1006l\u001b[?1003l\u001b[?1000l"
ENABLE_BRACKETED_PASTE = "\u001b[?2004h"
DISABLE_BRACKETED_PASTE = "\u001b[?2004l"
ENABLE_FOCUS_TRACKING = "\u001b[?1004h"
DISABLE_FOCUS_TRACKING = "\u001b[?1004l"
ENABLE_SYNC_OUTPUT = "\u001b[?2026h"
DISABLE_SYNC_OUTPUT = "\u001b[?2026l"

_DUMB_TERMS = frozenset({"dumb", "linux"})
_vt_enabled = False


def _enable_windows_vt_processing() -> None:
    """Turn on ANSI/VT processing on Windows 10+ consoles."""

    global _vt_enabled
    if _vt_enabled:
        return
    if sys.platform != "win32":
        _vt_enabled = True
        return
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        vt_out = 0x0004
        processed = 0x0001
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | vt_out | processed)
        h_in = kernel32.GetStdHandle(-10)
        mode_in = wintypes.DWORD()
        if kernel32.GetConsoleMode(h_in, ctypes.byref(mode_in)):
            kernel32.SetConsoleMode(h_in, mode_in.value | 0x0200)
        _vt_enabled = True
    except Exception:
        _vt_enabled = True


def hide_cursor() -> None:
    _enable_windows_vt_processing()
    sys.stdout.write("\u001b[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\u001b[?25h")
    sys.stdout.flush()


def is_dumb_terminal() -> bool:
    """Pipes and classic consoles cannot host the alternate screen."""

    if not sys.stdout.isatty():
        return True
    return os.environ.get("TERM", "") in _DUMB_TERMS


def enter_alternate_screen() -> None:
    _enable_windows_vt_processing()
    if is_dumb_terminal():
        return
    sys.stdout.write(
        DISABLE_MOUSE_TRACKING
        + ENTER_ALT_SCREEN
        + ERASE_SCREEN_AND_HOME
        + ENABLE_MOUSE_TRACKING
        + ENABLE_BRACKETED_PASTE
        + ENABLE_FOCUS_TRACKING
        + ENABLE_SYNC_OUTPUT
    )
    sys.stdout.flush()


def exit_alternate_screen() -> None:
    if is_dumb_terminal():
        return
    sys.stdout.write(
        DISABLE_MOUSE_TRACKING
        + EXIT_ALT_SCREEN
        + DISABLE_BRACKETED_PASTE
        + DISABLE_FOCUS_TRACKING
        + DISABLE_SYNC_OUTPUT
    )
    sys.stdout.flush()


def clear_screen() -> None:
    sys.stdout.write("\u001b[H\u001b[J")
    sys.stdout.flush()
