from microcode.mock_model import MockModelAdapter
from microcode.repl import apply_local_command
from microcode.tooling import ToolRegistry
from microcode.tui.app import should_use_tui, submit_line, transcript_from_messages
from microcode.tui.chrome import render_banner, render_permission_prompt
from microcode.tui.commands import matching_slash_commands
from microcode.tui.input_parser import KeyEvent, TextEvent, parse_input_chunk
from microcode.tui.keys import handle_event
from microcode.tui.screen import is_dumb_terminal
from microcode.tui.state import PendingPrompt, ScreenState, TuiContext
from microcode.tui.transcript import render_transcript
from microcode.tui.types import TranscriptEntry


def _ctx() -> TuiContext:
    return TuiContext(
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
    )


def test_parse_enter_backspace_and_arrows() -> None:
    parsed = parse_input_chunk("ab\x7f")
    kinds = [(type(event).__name__, getattr(event, "name", getattr(event, "text", ""))) for event in parsed.events]
    assert ("TextEvent", "a") in kinds
    assert ("TextEvent", "b") in kinds
    assert ("KeyEvent", "backspace") in kinds
    assert parse_input_chunk("\r").events[0].name == "return"

    arrows = parse_input_chunk("\x1b[A\x1b[B\x1b[C\x1b[D")
    assert [event.name for event in arrows.events] == ["up", "down", "right", "left"]


def test_parse_ctrl_c() -> None:
    parsed = parse_input_chunk("\x03")
    assert parsed.events == [KeyEvent(name="c", ctrl=True, meta=False)]


def test_parse_incomplete_escape_is_held() -> None:
    parsed = parse_input_chunk("\x1b[")
    assert parsed.events == []
    assert parsed.rest == "\x1b["


def test_render_banner_includes_model_and_cwd() -> None:
    rendered = render_banner(
        model="gpt-test",
        cwd="/tmp/demo",
        session_id="abc12345",
        mode="ask",
        message_count=4,
        width=80,
    )
    assert "MicroCode" in rendered
    assert "gpt-test" in rendered
    assert "/tmp/demo" in rendered
    assert "ask" in rendered


def test_render_transcript_shows_tool_and_hides_collapsed_body() -> None:
    entries = [
        TranscriptEntry(id=1, kind="user", body="hi"),
        TranscriptEntry(
            id=2,
            kind="tool",
            body="full output here",
            tool_name="read_file",
            status="success",
            collapsed=True,
            collapsed_summary="short summary",
        ),
    ]
    rendered = render_transcript(entries, scroll_offset=0, width=80, height=8)
    assert "read_file" in rendered
    assert "short summary" in rendered
    assert "full output here" not in rendered


def test_render_permission_prompt_lists_choices() -> None:
    rendered = render_permission_prompt(
        PendingPrompt(kind="approve", summary="Need approval for notes.txt"),
        width=80,
    )
    assert "Need approval for notes.txt" in rendered
    assert "[y] yes once" in rendered


def test_matching_slash_commands_filters_prefix() -> None:
    names = [command.name for command in matching_slash_commands("/he")]
    assert "/help" in names
    assert matching_slash_commands("/help extra") == []
    assert matching_slash_commands("help") == []


def test_handle_event_types_and_submits_help() -> None:
    state = ScreenState()
    submitted: list[str] = []

    def submit(text: str) -> bool:
        submitted.append(text)
        return False

    handle_event(state, TextEvent(text="/help", ctrl=False, meta=False), submit, width=80, height=10)
    assert state.input == "/help"
    handle_event(
        state,
        KeyEvent(name="return", ctrl=False, meta=False),
        submit,
        width=80,
        height=10,
    )
    assert submitted == ["/help"]
    assert state.input == ""


def test_handle_event_prefix_completes_slash_command() -> None:
    state = ScreenState(input="/he", cursor_offset=3)
    submitted: list[str] = []
    handle_event(
        state,
        KeyEvent(name="return", ctrl=False, meta=False),
        lambda text: submitted.append(text) or False,
        width=80,
        height=10,
    )
    assert state.input == "/help"
    assert submitted == []


def test_handle_event_ctrl_c_exits() -> None:
    state = ScreenState()
    assert handle_event(
        state,
        KeyEvent(name="c", ctrl=True, meta=False),
        lambda _text: False,
        width=80,
        height=10,
    )


def test_should_use_tui_skips_once_and_pipes(monkeypatch) -> None:
    assert should_use_tui(once=True) is False
    assert should_use_tui(once=False, force=False) is False
    monkeypatch.setattr("microcode.tui.app.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("microcode.tui.app.sys.stdout.isatty", lambda: True)
    assert should_use_tui(once=False, force=None) is False


def test_is_dumb_terminal_false_on_windows_with_empty_term(monkeypatch) -> None:
    monkeypatch.setattr("microcode.tui.screen.sys.platform", "win32")
    monkeypatch.setattr("microcode.tui.screen.sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("TERM", raising=False)
    assert is_dumb_terminal() is False


def test_is_dumb_terminal_true_when_piped(monkeypatch) -> None:
    monkeypatch.setattr("microcode.tui.screen.sys.stdout.isatty", lambda: False)
    assert is_dumb_terminal() is True


def test_transcript_from_messages_maps_roles() -> None:
    entries = transcript_from_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "tool_result",
                "toolName": "read_file",
                "content": "file body",
                "isError": False,
            },
        ]
    )
    assert [entry.kind for entry in entries] == ["user", "assistant", "tool"]
    assert entries[2].collapsed is True
    assert entries[2].tool_name == "read_file"


def test_submit_line_help_and_collapse() -> None:
    ctx = _ctx()
    state = ScreenState()
    assert submit_line(ctx, state, "/help", lambda: None) is False
    assert any(entry.kind == "user" and entry.body == "/help" for entry in state.transcript)
    assert any("/exit" in entry.body for entry in state.transcript if entry.kind == "assistant")

    state.push("tool", "full output here", tool_name="read_file", status="success")
    assert submit_line(ctx, state, "/collapse", lambda: None) is False
    tools = [entry for entry in state.transcript if entry.kind == "tool"]
    assert tools[-1].collapsed is True


def test_submit_line_exit() -> None:
    ctx = _ctx()
    state = ScreenState()
    assert submit_line(ctx, state, "/exit", lambda: None) is True


def test_apply_local_command_help_and_unknown_passthrough() -> None:
    result = apply_local_command(
        "/help",
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
    )
    assert result.handled is True
    assert "/status" in result.output
    skipped = apply_local_command(
        "please list files",
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=[{"role": "system", "content": "sys"}],
        cwd=".",
    )
    assert skipped.handled is False
