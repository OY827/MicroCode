from __future__ import annotations

from typing import Callable

from microcode.tui.commands import matching_slash_commands
from microcode.tui.input_parser import KeyEvent, ParsedInputEvent, TextEvent, WheelEvent
from microcode.tui.state import ScreenState
from microcode.tui.transcript import max_scroll_offset

SubmitFn = Callable[[str], bool]


def history_up(state: ScreenState) -> bool:
    if not state.history:
        return False
    if state.history_index >= len(state.history):
        state.history_draft = state.input
        state.history_index = len(state.history)
    if state.history_index <= 0:
        return False
    state.history_index -= 1
    state.input = state.history[state.history_index]
    state.cursor_offset = len(state.input)
    return True


def history_down(state: ScreenState) -> bool:
    if state.history_index >= len(state.history):
        return False
    state.history_index += 1
    if state.history_index >= len(state.history):
        state.input = state.history_draft
    else:
        state.input = state.history[state.history_index]
    state.cursor_offset = len(state.input)
    return True


def word_left(text: str, cursor: int) -> int:
    if cursor <= 1:
        return 0
    index = cursor - 1
    while index > 0 and text[index].isspace():
        index -= 1
    while index > 0 and not text[index - 1].isspace():
        index -= 1
    return index


def word_right(text: str, cursor: int) -> int:
    length = len(text)
    if cursor >= length:
        return length
    index = cursor
    while index < length and not text[index].isspace():
        index += 1
    while index < length and text[index].isspace():
        index += 1
    return index


def scroll_transcript(state: ScreenState, delta: int, *, width: int, height: int) -> bool:
    maximum = max_scroll_offset(state.transcript, width, height)
    nxt = max(0, min(maximum, state.transcript_scroll_offset + delta))
    if nxt == state.transcript_scroll_offset:
        return False
    state.transcript_scroll_offset = nxt
    return True


def handle_event(
    state: ScreenState,
    event: ParsedInputEvent,
    submit: SubmitFn,
    *,
    width: int,
    height: int,
    on_approval: Callable[[str], None] | None = None,
    on_ask: Callable[[str], None] | None = None,
) -> bool:
    """Apply one keyboard event. Return True when the session should exit."""

    if isinstance(event, KeyEvent) and event.ctrl and event.name == "c":
        return True
    if isinstance(event, TextEvent) and event.ctrl and event.text == "c":
        return True

    if state.pending is not None:
        return _handle_pending(state, event, on_approval=on_approval, on_ask=on_ask)

    if isinstance(event, WheelEvent):
        scroll_transcript(state, 3 if event.direction == "up" else -3, width=width, height=height)
        return False

    commands = matching_slash_commands(state.input)
    if isinstance(event, KeyEvent):
        return _handle_key(state, event, commands, submit, width=width, height=height)
    if isinstance(event, TextEvent):
        return _handle_text(state, event, submit)
    return False


def _handle_pending(
    state: ScreenState,
    event: ParsedInputEvent,
    *,
    on_approval: Callable[[str], None] | None,
    on_ask: Callable[[str], None] | None,
) -> bool:
    pending = state.pending
    if pending is None:
        return False
    if pending.kind == "ask":
        if isinstance(event, KeyEvent):
            if event.name == "escape":
                if on_ask:
                    on_ask("")
                return False
            if event.name == "return":
                if on_ask:
                    on_ask(pending.answer)
                return False
            if event.name == "backspace" and pending.answer:
                pending.answer = pending.answer[:-1]
                return False
        if isinstance(event, TextEvent) and not event.ctrl:
            pending.answer += event.text
        return False

    if pending.kind == "edit":
        if isinstance(event, KeyEvent):
            if event.name == "escape":
                pending.kind = "approve"
                pending.answer = ""
                return False
            if event.name == "return":
                if on_approval:
                    on_approval("edit")
                return False
            if event.name == "backspace" and pending.answer:
                pending.answer = pending.answer[:-1]
                return False
        if isinstance(event, TextEvent) and not event.ctrl:
            pending.answer += event.text
        return False

    if isinstance(event, TextEvent) and not event.ctrl and event.text.lower() == "e" and pending.proposed:
        pending.kind = "edit"
        pending.answer = pending.proposed
        return False

    decision = None
    if isinstance(event, KeyEvent):
        if event.name in {"escape"}:
            decision = "no"
        elif event.name == "return":
            decision = "yes"
        elif event.name == "y":
            decision = "yes"
        elif event.name == "n":
            decision = "no"
        elif event.name == "a":
            decision = "always"
    if isinstance(event, TextEvent) and not event.ctrl:
        lowered = event.text.lower()
        if lowered in {"y", "n", "a"}:
            decision = {"y": "yes", "n": "no", "a": "always"}[lowered]
    if decision is not None and on_approval:
        on_approval(decision)
    return False


def _handle_key(
    state: ScreenState,
    event: KeyEvent,
    commands: list,
    submit: SubmitFn,
    *,
    width: int,
    height: int,
) -> bool:
    if event.name == "return":
        if commands:
            selected = commands[min(state.selected_slash_index, len(commands) - 1)]
            typed = state.input.strip().lower()
            if typed not in {selected.name, selected.usage.strip().lower()}:
                state.input = selected.usage
                state.cursor_offset = len(state.input)
                state.selected_slash_index = 0
                return False
        submitted = state.input
        state.input = ""
        state.cursor_offset = 0
        state.selected_slash_index = 0
        if not submitted.strip() or state.is_busy:
            return False
        return submit(submitted)

    if event.name == "tab" and commands:
        selected = commands[min(state.selected_slash_index, len(commands) - 1)]
        state.input = selected.usage
        state.cursor_offset = len(state.input)
        state.selected_slash_index = 0
        return False

    if event.name == "pageup":
        scroll_transcript(state, 8, width=width, height=height)
        return False
    if event.name == "pagedown":
        scroll_transcript(state, -8, width=width, height=height)
        return False
    if event.name == "home" and event.ctrl:
        state.transcript_scroll_offset = max_scroll_offset(state.transcript, width, height)
        return False
    if event.name == "end" and event.ctrl:
        state.transcript_scroll_offset = 0
        return False

    if event.name == "up":
        if commands:
            state.selected_slash_index = (state.selected_slash_index - 1) % len(commands)
        else:
            history_up(state)
        return False
    if event.name == "down":
        if commands:
            state.selected_slash_index = (state.selected_slash_index + 1) % len(commands)
        else:
            history_down(state)
        return False

    if event.name == "backspace" and state.cursor_offset > 0:
        state.input = state.input[: state.cursor_offset - 1] + state.input[state.cursor_offset :]
        state.cursor_offset -= 1
        state.selected_slash_index = 0
        return False
    if event.name == "delete" and state.cursor_offset < len(state.input):
        state.input = state.input[: state.cursor_offset] + state.input[state.cursor_offset + 1 :]
        state.selected_slash_index = 0
        return False
    if event.name == "home":
        state.cursor_offset = 0
        return False
    if event.name == "end":
        state.cursor_offset = len(state.input)
        return False
    if event.name == "left":
        if event.ctrl:
            state.cursor_offset = word_left(state.input, state.cursor_offset)
        else:
            state.cursor_offset = max(0, state.cursor_offset - 1)
        return False
    if event.name == "right":
        if event.ctrl:
            state.cursor_offset = word_right(state.input, state.cursor_offset)
        else:
            state.cursor_offset = min(len(state.input), state.cursor_offset + 1)
        return False
    if event.name == "escape":
        state.input = ""
        state.cursor_offset = 0
        state.selected_slash_index = 0
        return False
    if event.ctrl and event.name == "u":
        state.input = ""
        state.cursor_offset = 0
        state.selected_slash_index = 0
        return False
    if event.ctrl and event.name == "w":
        target = word_left(state.input, state.cursor_offset)
        state.input = state.input[:target] + state.input[state.cursor_offset :]
        state.cursor_offset = target
        state.selected_slash_index = 0
        return False
    if event.ctrl and event.name == "k":
        state.input = state.input[: state.cursor_offset]
        state.selected_slash_index = 0
        return False
    if event.ctrl and event.name == "a":
        state.cursor_offset = 0
        return False
    if event.ctrl and event.name == "e":
        state.cursor_offset = len(state.input)
        return False
    if event.ctrl and event.name == "p":
        history_up(state)
        return False
    if event.ctrl and event.name == "n":
        history_down(state)
        return False
    return False


def _handle_text(state: ScreenState, event: TextEvent, submit: SubmitFn) -> bool:
    if event.ctrl:
        return False
    if not event.text:
        return False
    state.input = state.input[: state.cursor_offset] + event.text + state.input[state.cursor_offset :]
    state.cursor_offset += len(event.text)
    state.selected_slash_index = 0
    state.history_index = len(state.history)
    return False
