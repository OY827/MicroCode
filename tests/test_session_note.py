from pathlib import Path

from microcode.checkpoint import WriteCheckpoint, remember_write_group
from microcode.memory import ProjectMemory
from microcode.repl import run_repl
from microcode.session import SessionStore
from microcode.session_note import build_session_note, finish_session_memory
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage


class SilentModel:
    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        return AgentStep(type="assistant", content="ok")


def test_build_note_from_ask_and_files(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    checkpoint = WriteCheckpoint()
    checkpoint.record("app.py", existed=True, content="old")
    checkpoint.record("notes.txt", existed=False)
    remember_write_group(session, checkpoint)
    text = build_session_note(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the timeout in app.py"},
            {"role": "assistant", "content": "done"},
        ],
        session,
    )
    assert f"Last chat ({session.id})" in text
    assert "fix the timeout in app.py" in text
    assert "app.py" in text
    assert "notes.txt" in text


def test_build_note_skips_empty() -> None:
    assert build_session_note([{"role": "system", "content": "sys"}]) == ""


def test_finish_writes_then_updates_same_session(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [])
    first = finish_session_memory(
        memory,
        messages=[{"role": "user", "content": "first task"}],
        session=session,
    )
    assert "Remembered session note" in first
    notes = memory.manager.all_entries()
    assert len(notes) == 1
    assert notes[0].tier.value == "short_term"
    assert "first task" in notes[0].content

    second = finish_session_memory(
        memory,
        messages=[{"role": "user", "content": "second task"}],
        session=session,
    )
    assert "Updated session note" in second
    notes = memory.manager.all_entries()
    assert len(notes) == 1
    assert "second task" in notes[0].content
    assert "first task" not in notes[0].content


def test_session_note_can_be_turned_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_SESSION_NOTE", "0")
    memory = ProjectMemory(tmp_path)
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [])
    report = finish_session_memory(
        memory,
        messages=[{"role": "user", "content": "secret task"}],
        session=session,
    )
    assert "session note" not in report
    assert memory.manager.all_entries() == []


def test_repl_exit_writes_session_note(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    lines = iter(["please use pytest", "/exit"])
    run_repl(
        model=SilentModel(),
        tools=ToolRegistry([]),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        memory=memory,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    texts = [entry.content for entry in memory.manager.all_entries()]
    assert any("please use pytest" in text for text in texts)
