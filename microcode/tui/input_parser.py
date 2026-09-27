from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Union

_SGR_MOUSE_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_CSI_CURSOR_RE = re.compile(r"^\x1b\[(?:1;(\d+))?([A-DF-H])")
_CSI_TILDE_RE = re.compile(r"^\x1b\[(\d+)(?:;(\d+))?~")
_SS3_RE = re.compile(r"^\x1bO([A-DF-H])")
_ESC_CHAR_RE = re.compile(r"^\x1b([^\x1b\[O])")


@dataclass(frozen=True)
class KeyEvent:
    name: str
    ctrl: bool
    meta: bool
    kind: str = "key"


@dataclass(frozen=True)
class TextEvent:
    text: str
    ctrl: bool
    meta: bool
    kind: str = "text"


@dataclass(frozen=True)
class WheelEvent:
    direction: Literal["up", "down"]
    kind: str = "wheel"


ParsedInputEvent = Union[KeyEvent, TextEvent, WheelEvent]


@dataclass(frozen=True)
class ParseResult:
    events: list[ParsedInputEvent]
    rest: str


CTRL_CHAR_TO_NAME: dict[str, str] = {
    "\x01": "a",
    "\x03": "c",
    "\x05": "e",
    "\x0e": "n",
    "\x0f": "o",
    "\x10": "p",
    "\x15": "u",
    "\x17": "w",
    "\x0b": "k",
}


def _is_multiline_paste_chunk(chunk: str) -> bool:
    has_newline = False
    has_other = False
    for char in chunk:
        if char in {"\r", "\n"}:
            has_newline = True
        else:
            has_other = True
        if has_newline and has_other:
            return True
    return False


def maybe_need_more_for_escape_sequence(chunk: str) -> bool:
    if not chunk or chunk[0] != "\x1b":
        return False
    if len(chunk) == 1:
        return True
    if chunk[1] == "[":
        if len(chunk) >= 3 and chunk[2] == "<":
            return not any(char in "Mm" for char in chunk[3:])
        if len(chunk) >= 3 and chunk[2] == "M":
            return len(chunk) < 6
        for index in range(2, len(chunk)):
            char = chunk[index]
            if "A" <= char <= "Z" or "a" <= char <= "z" or char == "~":
                return False
            if char not in "0123456789;?":
                return False
        return True
    if chunk[1] == "O":
        return len(chunk) < 3
    return False


def parse_escape_sequence(chunk: str) -> tuple[ParsedInputEvent | None, int]:
    if not chunk or chunk[0] != "\x1b":
        return None, 0
    if len(chunk) == 1:
        return KeyEvent(name="escape", ctrl=False, meta=False), 1

    sgr_match = _SGR_MOUSE_RE.match(chunk)
    if sgr_match:
        button = int(sgr_match.group(1))
        if (button & 0x43) == 0x40:
            return WheelEvent(direction="up"), sgr_match.end()
        if (button & 0x43) == 0x41:
            return WheelEvent(direction="down"), sgr_match.end()
        return None, sgr_match.end()

    if chunk.startswith("\x1b[M") and len(chunk) >= 6:
        button = ord(chunk[3])
        if (button & 0x43) == 0x40:
            return WheelEvent(direction="up"), 6
        if (button & 0x43) == 0x41:
            return WheelEvent(direction="down"), 6
        return None, 6

    csi_cursor_match = _CSI_CURSOR_RE.match(chunk)
    if csi_cursor_match:
        mod_str = csi_cursor_match.group(1)
        key_char = csi_cursor_match.group(2)
        mod = int(mod_str) if mod_str else 1
        ctrl = bool((mod - 1) & 4)
        meta = bool((mod - 1) & 2)
        names = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
        return KeyEvent(name=names[key_char], ctrl=ctrl, meta=meta), csi_cursor_match.end()

    csi_tilde_match = _CSI_TILDE_RE.match(chunk)
    if csi_tilde_match:
        number = int(csi_tilde_match.group(1))
        mod_str = csi_tilde_match.group(2)
        mod = int(mod_str) if mod_str else 1
        ctrl = bool((mod - 1) & 4)
        meta = bool((mod - 1) & 2)
        names = {1: "home", 3: "delete", 4: "end", 5: "pageup", 6: "pagedown", 7: "home", 8: "end"}
        if number in names:
            return KeyEvent(name=names[number], ctrl=ctrl, meta=meta), csi_tilde_match.end()
        return None, csi_tilde_match.end()

    ss3_match = _SS3_RE.match(chunk)
    if ss3_match:
        names = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
        return KeyEvent(name=names[ss3_match.group(1)], ctrl=False, meta=False), ss3_match.end()

    if chunk.startswith("\x1b\t"):
        return KeyEvent(name="tab", ctrl=False, meta=True), 2

    esc_char_match = _ESC_CHAR_RE.match(chunk)
    if esc_char_match:
        return TextEvent(text=esc_char_match.group(1), ctrl=False, meta=True), 2

    return KeyEvent(name="escape", ctrl=False, meta=False), 1


def parse_input_chunk(chunk: str, incoming_chunk: str | None = None) -> ParseResult:
    treat_newlines_as_text = _is_multiline_paste_chunk(
        incoming_chunk if incoming_chunk is not None else chunk
    )
    events: list[ParsedInputEvent] = []
    index = 0
    while index < len(chunk):
        if maybe_need_more_for_escape_sequence(chunk[index:]):
            break
        char = chunk[index]
        if char == "\x1b":
            if chunk[index : index + 3] == "\x1b[I":
                events.append(KeyEvent(name="focus_in", ctrl=False, meta=False))
                index += 3
                continue
            if chunk[index : index + 3] == "\x1b[O":
                events.append(KeyEvent(name="focus_out", ctrl=False, meta=False))
                index += 3
                continue
            if chunk[index : index + 6] == "\x1b[200~" and not maybe_need_more_for_escape_sequence(
                chunk[index + 6 :]
            ):
                index += 6
                paste_end = chunk.find("\x1b[201~", index)
                if paste_end >= 0:
                    paste_text = "".join(
                        item for item in chunk[index:paste_end] if item.isprintable() or item in "\n\t"
                    )
                    events.append(TextEvent(text=paste_text, ctrl=False, meta=False))
                    index = paste_end + 6
                    continue
                break
            event, consumed = parse_escape_sequence(chunk[index:])
            if event:
                events.append(event)
            index += consumed
            continue
        if char in {"\r", "\n"}:
            if treat_newlines_as_text:
                events.append(TextEvent(text="\n", ctrl=False, meta=False))
            else:
                events.append(KeyEvent(name="return", ctrl=False, meta=False))
            if (char == "\r" and index + 1 < len(chunk) and chunk[index + 1] == "\n") or (
                char == "\n" and index + 1 < len(chunk) and chunk[index + 1] == "\r"
            ):
                index += 2
            else:
                index += 1
            continue
        if char == "\t":
            events.append(KeyEvent(name="tab", ctrl=False, meta=False))
            index += 1
            continue
        if char in {"\x7f", "\x08"}:
            events.append(KeyEvent(name="backspace", ctrl=False, meta=False))
            index += 1
            continue
        if "\x01" <= char <= "\x1a":
            if char in CTRL_CHAR_TO_NAME:
                events.append(KeyEvent(name=CTRL_CHAR_TO_NAME[char], ctrl=True, meta=False))
            index += 1
            continue
        events.append(TextEvent(text=char, ctrl=False, meta=False))
        index += 1
    return ParseResult(events=events, rest=chunk[index:])
