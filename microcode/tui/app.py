from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable

from microcode.models import model_label
from microcode.repl import apply_local_command, handle_local_command, run_user_turn
from microcode.session import Session
from microcode.session_note import finish_session_memory
from microcode.shortcuts import execute_tool_shortcut, parse_tool_shortcut
from microcode.tooling import WriteDecision
from microcode.tui.chrome import (
    invalidate_terminal_size_cache,
    render_banner,
    render_footer,
    render_input_prompt,
    render_permission_prompt,
    render_slash_menu,
    terminal_size,
)
from microcode.tui.commands import matching_slash_commands
from microcode.tui.input_parser import parse_input_chunk
from microcode.tui.keys import handle_event
from microcode.tui.screen import (
    ENABLE_SYNC_OUTPUT,
    DISABLE_SYNC_OUTPUT,
    enter_alternate_screen,
    exit_alternate_screen,
    hide_cursor,
    is_dumb_terminal,
    show_cursor,
)
from microcode.tui.state import PendingPrompt, ScreenState, TuiContext
from microcode.tui.transcript import render_transcript
from microcode.types import ChatMessage

_WIN_SCANCODE_TO_ANSI: dict[int, str] = {
    72: "\x1b[A",
    80: "\x1b[B",
    77: "\x1b[C",
    75: "\x1b[D",
    71: "\x1b[H",
    79: "\x1b[F",
    73: "\x1b[5~",
    81: "\x1b[6~",
    83: "\x1b[3~",
}


def should_use_tui(*, once: bool = False, force: bool | None = None) -> bool:
    if once:
        return False
    if force is False:
        return False
    if force is True:
        return True
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty() and not is_dumb_terminal())
    except Exception:
        return False


def transcript_from_messages(messages: list[ChatMessage]) -> list:
    from microcode.tui.types import TranscriptEntry

    entries: list[TranscriptEntry] = []
    next_id = 1
    for message in messages:
        role = message.get("role")
        if role == "user":
            entries.append(
                TranscriptEntry(id=next_id, kind="user", body=str(message.get("content", "")))
            )
        elif role == "assistant":
            entries.append(
                TranscriptEntry(
                    id=next_id, kind="assistant", body=str(message.get("content", ""))
                )
            )
        elif role == "tool_result":
            is_error = bool(message.get("isError"))
            output = str(message.get("content", ""))
            entries.append(
                TranscriptEntry(
                    id=next_id,
                    kind="tool",
                    body=output,
                    tool_name=str(message.get("toolName") or "tool"),
                    status="error" if is_error else "success",
                    collapsed=True,
                    collapsed_summary=(output[:80] + "…") if len(output) > 80 else output,
                )
            )
        else:
            continue
        next_id += 1
    return entries


class _RawModeContext:
    def __init__(self) -> None:
        self._old_settings: Any = None
        self._old_cp: int | None = None
        self._old_sigwinch: Any = None

    def __enter__(self) -> _RawModeContext:
        if sys.platform == "win32":
            from microcode.tui.screen import _enable_windows_vt_processing

            _enable_windows_vt_processing()
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                self._old_cp = kernel32.GetConsoleOutputCP()
                kernel32.SetConsoleOutputCP(65001)
            except Exception:
                pass
            return self
        import signal
        import termios

        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        new[2] &= ~(termios.CSIZE | termios.PARENB)
        new[2] |= termios.CS8
        new[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSAFLUSH, new)
        try:
            def _on_resize(_signum, _frame):
                invalidate_terminal_size_cache()

            self._old_sigwinch = signal.signal(signal.SIGWINCH, _on_resize)
        except (ValueError, AttributeError):
            pass
        return self

    def __exit__(self, *_: Any) -> None:
        if sys.platform == "win32":
            if self._old_cp is not None:
                try:
                    import ctypes

                    ctypes.windll.kernel32.SetConsoleOutputCP(self._old_cp)  # type: ignore[attr-defined]
                except Exception:
                    pass
            return
        if self._old_settings is not None:
            import signal
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            if self._old_sigwinch is not None:
                try:
                    signal.signal(signal.SIGWINCH, self._old_sigwinch)
                except Exception:
                    pass


def _win_read_one_key() -> str:
    import msvcrt

    if not msvcrt.kbhit():
        return ""
    char = msvcrt.getwch()
    if char in {"\x00", "\xe0"}:
        if msvcrt.kbhit():
            scan = ord(msvcrt.getwch())
        else:
            return "\x1b"
        return _WIN_SCANCODE_TO_ANSI.get(scan, "")
    return char


def _read_raw_chunk() -> str:
    if sys.platform == "win32":
        result = ""
        while True:
            char = _win_read_one_key()
            if not char:
                break
            result += char
        return result
    import select

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], 0.05)
    if not ready:
        return ""
    data = os.read(fd, 4096)
    while True:
        ready2, _, _ = select.select([fd], [], [], 0)
        if not ready2:
            break
        more = os.read(fd, 4096)
        if not more:
            break
        data += more
    return data.decode("utf-8", errors="replace") if data else ""


class _ThrottledRenderer:
    def __init__(self, draw: Callable[[], None], min_interval: float = 0.016) -> None:
        self._draw = draw
        self._min = min_interval
        self._pending = False
        self._last = 0.0
        self._lock = threading.Lock()

    def request(self) -> None:
        with self._lock:
            self._pending = True

    def flush(self, *, force: bool = False) -> None:
        with self._lock:
            if not self._pending and not force:
                return
            now = time.monotonic()
            if not force and now - self._last < self._min:
                return
            self._pending = False
            self._last = now
        self._draw()


def _session_label(session: Session | None) -> str:
    if session is None:
        return "-"
    return session.id[:8]


def render_screen(ctx: TuiContext, state: ScreenState) -> None:
    cols, rows = terminal_size()
    cols = max(40, cols)
    rows = max(12, rows)
    header = render_banner(
        model=model_label(ctx.model) or "mock",
        cwd=ctx.cwd,
        session_id=_session_label(ctx.session),
        mode=ctx.session.permission_mode if ctx.session is not None else "ask",
        message_count=len(ctx.messages),
        width=cols,
    ).splitlines()
    footer = render_footer(state.status, width=cols).splitlines()
    overlay: list[str] = []
    if state.pending is not None:
        overlay = render_permission_prompt(state.pending, width=cols).splitlines()
        prompt: list[str] = []
    else:
        prompt = [render_input_prompt(state.input, state.cursor_offset, width=cols)]
        commands = matching_slash_commands(state.input)
        if commands:
            selected = min(state.selected_slash_index, len(commands) - 1)
            prompt.extend(render_slash_menu(commands, selected, width=cols).splitlines())
    used = len(header) + len(footer) + len(overlay) + len(prompt)
    transcript_height = max(3, rows - used)
    transcript = render_transcript(
        state.transcript,
        scroll_offset=state.transcript_scroll_offset,
        width=cols,
        height=transcript_height,
    ).splitlines()
    lines = header + transcript + overlay + prompt + footer
    if len(lines) < rows:
        lines.extend([""] * (rows - len(lines)))
    elif len(lines) > rows:
        lines = lines[:rows]
    sys.stdout.write(ENABLE_SYNC_OUTPUT + "\u001b[H" + "\n".join(lines) + DISABLE_SYNC_OUTPUT)
    sys.stdout.flush()


def _remember_history(state: ScreenState, text: str) -> None:
    if not state.history or state.history[-1] != text:
        state.history.append(text)
    state.history_index = len(state.history)
    state.history_draft = ""


def submit_line(ctx: TuiContext, state: ScreenState, text: str, rerender: Callable[[], None]) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    _remember_history(state, stripped)
    local_name = handle_local_command(stripped)
    if local_name == "exit":
        return True
    state.push("user", stripped)

    if stripped.lower() in {"/collapse"}:
        collapsed = 0
        for entry in state.transcript:
            if entry.kind == "tool" and not entry.collapsed:
                entry.collapsed = True
                entry.collapsed_summary = entry.collapsed_summary or "output collapsed"
                collapsed += 1
        state.push("assistant", f"Collapsed {collapsed} tool-output block(s).")
        return False

    if local_name is not None:
        result = apply_local_command(
            stripped,
            model=ctx.model,
            tools=ctx.tools,
            messages=ctx.messages,
            cwd=ctx.cwd,
            session=ctx.session,
            store=ctx.store,
            memory=ctx.memory,
            system_prompt=ctx.system_prompt,
            mcp_status=ctx.mcp_status,
            permissions=ctx.permissions,
            todos=ctx.todos,
            jobs=ctx.jobs,
            usage=ctx.usage,
        )
        if result.messages is not None:
            ctx.messages = result.messages
        if result.model is not None:
            ctx.model = result.model
        if result.output:
            state.push("assistant", result.output.rstrip("\n"))
        return False

    shortcut = parse_tool_shortcut(stripped)
    if isinstance(shortcut, str):
        state.push("assistant", shortcut)
        return False
    if shortcut is not None:
        _start_worker(ctx, state, rerender, lambda: _run_shortcut(ctx, state, shortcut, rerender))
        return False

    _start_worker(ctx, state, rerender, lambda: _run_turn(ctx, state, stripped, rerender))
    return False


def _start_worker(
    ctx: TuiContext,
    state: ScreenState,
    rerender: Callable[[], None],
    work: Callable[[], None],
) -> None:
    state.is_busy = True
    state.status = "Working..."
    state.agent_done = False
    state.agent_error = None
    rerender()

    def run() -> None:
        try:
            work()
        except Exception as error:  # noqa: BLE001
            state.agent_error = f"{type(error).__name__}: {error}"
            state.push("assistant", state.agent_error)
        finally:
            state.is_busy = False
            state.active_tool = None
            state.status = None
            state.stream_entry_id = None
            state.agent_done = True
            rerender()

    threading.Thread(target=run, daemon=True).start()


def _bind_prompts(state: ScreenState, rerender: Callable[[], None]) -> tuple[Callable, Callable, Callable]:
    approval_event = threading.Event()
    approval_box: dict[str, str] = {}
    ask_event = threading.Event()
    ask_box: dict[str, str] = {}

    def on_approve(summary: str):
        state.pending = PendingPrompt(kind="approve", summary=summary)
        approval_box.clear()
        approval_event.clear()
        rerender()
        approval_event.wait()
        state.pending = None
        rerender()
        decision = approval_box.get("decision", "no")
        if decision == "always":
            return "always"
        return decision == "yes"

    def on_ask(question: str) -> str:
        state.pending = PendingPrompt(kind="ask", summary=question)
        ask_box.clear()
        ask_event.clear()
        rerender()
        ask_event.wait()
        state.pending = None
        rerender()
        return ask_box.get("answer", "")

    def resolve_approval(decision: str) -> None:
        approval_box["decision"] = decision
        if decision == "edit" and state.pending is not None:
            approval_box["content"] = state.pending.answer
        approval_event.set()

    def resolve_ask(answer: str) -> None:
        ask_box["answer"] = answer
        ask_event.set()

    def on_revise(path: str, proposed: str, preview: str) -> WriteDecision:
        state.pending = PendingPrompt(
            kind="approve",
            summary=f"Write {path}?\n{preview}",
            proposed=proposed,
        )
        approval_box.clear()
        approval_event.clear()
        rerender()
        approval_event.wait()
        state.pending = None
        rerender()
        decision = approval_box.get("decision", "no")
        if decision == "edit":
            return WriteDecision(allow=True, content=str(approval_box.get("content", proposed)))
        if decision == "always":
            return WriteDecision(allow=True, remember=True)
        if decision == "yes":
            return WriteDecision(allow=True)
        return WriteDecision(allow=False)

    state.resolve_approval = resolve_approval
    state.resolve_ask = resolve_ask
    return on_approve, on_ask, on_revise


def _tool_callbacks(state: ScreenState, rerender: Callable[[], None]) -> dict:
    pending_tools: dict[str, list[int]] = {}

    def on_tool_start(name: str, args: dict) -> None:
        preview = str(args)
        if len(preview) > 120:
            preview = preview[:120] + "..."
        entry_id = state.push("tool", preview, tool_name=name, status="running")
        pending_tools.setdefault(name, []).append(entry_id)
        state.active_tool = name
        state.status = f"Running {name}..."
        rerender()

    def on_tool_result(name: str, output: str, is_error: bool) -> None:
        ids = pending_tools.get(name) or []
        entry_id = ids.pop(0) if ids else None
        entry = state.find(entry_id) if entry_id is not None else None
        if entry is not None:
            entry.body = output
            entry.status = "error" if is_error else "success"
            entry.collapsed = True
            summary = output.strip().splitlines()[0] if output.strip() else ("error" if is_error else "ok")
            entry.collapsed_summary = summary[:80]
        state.active_tool = None
        state.status = "Thinking..."
        rerender()

    def on_text_delta(chunk: str) -> None:
        if not chunk or chunk == "\n":
            if chunk == "\n":
                state.stream_entry_id = None
            return
        if state.stream_entry_id is None:
            state.stream_entry_id = state.push("assistant", chunk)
        else:
            entry = state.find(state.stream_entry_id)
            if entry is not None:
                entry.body += chunk
        rerender()

    def on_assistant_message(text: str) -> None:
        if state.stream_entry_id is not None:
            entry = state.find(state.stream_entry_id)
            if entry is not None:
                entry.body = text
            state.stream_entry_id = None
        else:
            state.push("assistant", text)
        rerender()

    def on_write_preview(path: str, diff: str) -> None:
        preview = diff if len(diff) <= 400 else diff[:400] + "\n..."
        state.push("progress", f"write preview {path}\n{preview}")
        rerender()

    def on_compact(report) -> None:
        state.push("progress", report.summary())
        rerender()

    return {
        "on_tool_start": on_tool_start,
        "on_tool_result": on_tool_result,
        "on_text_delta": on_text_delta,
        "on_assistant_message": on_assistant_message,
        "on_write_preview": on_write_preview,
        "on_compact": on_compact,
    }


def _run_shortcut(ctx: TuiContext, state: ScreenState, shortcut, rerender: Callable[[], None]) -> None:
    on_approve, _on_ask, on_revise = _bind_prompts(state, rerender)
    callbacks = _tool_callbacks(state, rerender)
    state.push("tool", str(shortcut.input_data), tool_name=shortcut.tool_name, status="running")
    entry_id = state.transcript[-1].id
    result = execute_tool_shortcut(
        shortcut,
        tools=ctx.tools,
        cwd=ctx.cwd,
        on_write_preview=callbacks["on_write_preview"],
        on_approve=on_approve,
        on_revise_write=on_revise,
        session=ctx.session,
        store=ctx.store,
        permissions=ctx.permissions,
        jobs=ctx.jobs,
    )
    entry = state.find(entry_id)
    if entry is not None:
        entry.body = result.output
        entry.status = "success" if result.ok else "error"
        entry.collapsed = True
        entry.collapsed_summary = ("ok" if result.ok else "error") + ": " + result.output[:60]


def _run_turn(ctx: TuiContext, state: ScreenState, text: str, rerender: Callable[[], None]) -> None:
    on_approve, on_ask, on_revise = _bind_prompts(state, rerender)
    callbacks = _tool_callbacks(state, rerender)
    ctx.messages = run_user_turn(
        model=ctx.model,
        tools=ctx.tools,
        messages=ctx.messages,
        user_text=text,
        cwd=ctx.cwd,
        session=ctx.session,
        store=ctx.store,
        permissions=ctx.permissions,
        todos=ctx.todos,
        jobs=ctx.jobs,
        system_prompt=ctx.system_prompt,
        memory=ctx.memory,
        usage=ctx.usage,
        on_approve=on_approve,
        on_revise_write=on_revise,
        on_ask_user=on_ask,
        **callbacks,
    )


def run_tui(
    *,
    ctx: TuiContext,
    initial_prompt: str | None = None,
) -> list[ChatMessage]:
    state = ScreenState()
    restored = transcript_from_messages(ctx.messages)
    if restored:
        state.transcript = restored
        state.next_entry_id = restored[-1].id + 1
    else:
        state.push("progress", "Interactive session. Type /help or a task. Ctrl+C leaves.")

    def draw() -> None:
        render_screen(ctx, state)

    throttled = _ThrottledRenderer(draw)
    remainder = ""

    def rerender() -> None:
        throttled.request()

    def submit(text: str) -> bool:
        return submit_line(ctx, state, text, rerender)

    enter_alternate_screen()
    hide_cursor()
    try:
        draw()
        if initial_prompt and initial_prompt.strip():
            if submit(initial_prompt.strip()):
                return ctx.messages
            draw()
        with _RawModeContext():
            while not state.should_exit:
                chunk = _read_raw_chunk()
                if not chunk:
                    throttled.flush()
                    if sys.platform == "win32":
                        time.sleep(0.02)
                    continue
                parsed = parse_input_chunk(remainder + chunk, incoming_chunk=chunk)
                remainder = parsed.rest
                cols, rows = terminal_size()
                transcript_height = max(3, rows - 10)
                for event in parsed.events:
                    should_exit = handle_event(
                        state,
                        event,
                        submit,
                        width=cols,
                        height=transcript_height,
                        on_approval=state.resolve_approval,
                        on_ask=state.resolve_ask,
                    )
                    if should_exit:
                        state.should_exit = True
                        break
                throttled.flush(force=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        show_cursor()
        exit_alternate_screen()
        if ctx.memory is not None:
            note = finish_session_memory(ctx.memory, messages=ctx.messages, session=ctx.session)
            print(note, file=sys.stderr)
        print("bye", file=sys.stderr)
    return ctx.messages
