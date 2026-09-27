from pathlib import Path

from microcode.session import SessionStore, format_replay


def test_replay_skips_system_and_lists_turns() -> None:
    text = format_replay(
        messages=[
            {"role": "system", "content": "long hidden instructions " * 20},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant_tool_call",
                "toolName": "list_files",
                "input": {"path": "."},
            },
            {"role": "tool_result", "toolName": "list_files", "content": "a.py\nb.py", "isError": False},
            {"role": "assistant", "content": "two files"},
        ]
    )
    assert "long hidden instructions" not in text
    assert "[1] you" in text
    assert "list files" in text
    assert "[2] tool list_files" in text
    assert "[3] result [ok]" in text
    assert "[4] assistant" in text
    assert "two files" in text


def test_replay_empty_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    text = format_replay(session=session)
    assert session.id in text
    assert "No conversation to replay yet." in text


def test_replay_keeps_session_header(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(
        str(tmp_path),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )
    session.model_name = "gpt-4o-mini"
    store.save(session)
    loaded = store.load(session.id)
    text = format_replay(messages=loaded.messages, session=loaded)
    assert f"replay: {session.id}" in text
    assert "model: gpt-4o-mini" in text
    assert "[1] you" in text
