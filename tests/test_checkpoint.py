from pathlib import Path

from microcode.checkpoint import WriteCheckpoint, undo_last_writes
from microcode.repl import run_repl, run_user_turn
from microcode.session import SessionStore
from microcode.tools import create_default_tool_registry
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _write_then_answer(path: str, content: str) -> list[AgentStep]:
    return [
        AgentStep(
            type="tool_calls",
            calls=[{"id": "1", "toolName": "write_file", "input": {"path": path, "content": content}}],
        ),
        AgentStep(type="assistant", content="done"),
    ]


def test_undo_restores_existing_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)

    messages = run_user_turn(
        model=ScriptedModel(_write_then_answer("notes.txt", "new\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="edit notes",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\n"
    report = undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert "restored notes.txt" in report
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"
    assert messages[-1]["content"] == "done"
    loaded = store.load(session.id)
    assert loaded.messages[-1]["content"] == "done"
    assert loaded.checkpoint == []


def test_undo_deletes_created_file(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)

    run_user_turn(
        model=ScriptedModel(_write_then_answer("hello.txt", "hello\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="create hello",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    assert (tmp_path / "hello.txt").is_file()
    report = undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert "deleted hello.txt" in report
    assert not (tmp_path / "hello.txt").exists()


def test_first_snapshot_wins_in_one_turn(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("v1\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = ScriptedModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "1",
                        "toolName": "write_file",
                        "input": {"path": "notes.txt", "content": "v2\n"},
                    }
                ],
            ),
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "2",
                        "toolName": "write_file",
                        "input": {"path": "notes.txt", "content": "v3\n"},
                    }
                ],
            ),
            AgentStep(type="assistant", content="done"),
        ]
    )

    run_user_turn(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="rewrite twice",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "v3\n"
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "v1\n"


def test_undo_only_last_write_turn(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A0\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B0\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    tools = create_default_tool_registry()

    messages = run_user_turn(
        model=ScriptedModel(_write_then_answer("a.txt", "A1\n")),
        tools=tools,
        messages=session.messages,
        user_text="edit a",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    run_user_turn(
        model=ScriptedModel(_write_then_answer("b.txt", "B1\n")),
        tools=tools,
        messages=messages,
        user_text="edit b",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A1\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B0\n"


def test_talk_only_turn_keeps_previous_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    tools = create_default_tool_registry()

    messages = run_user_turn(
        model=ScriptedModel(_write_then_answer("notes.txt", "new\n")),
        tools=tools,
        messages=session.messages,
        user_text="edit notes",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    run_user_turn(
        model=ScriptedModel([AgentStep(type="assistant", content="ok")]),
        tools=tools,
        messages=messages,
        user_text="just talk",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"


def test_rejected_write_does_not_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)

    run_user_turn(
        model=ScriptedModel(_write_then_answer("hello.txt", "hello\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="create hello",
        cwd=str(tmp_path),
        session=session,
        store=store,
        on_approve=lambda _summary: False,
    )

    assert not (tmp_path / "hello.txt").exists()
    assert undo_last_writes(session=session, store=store, cwd=str(tmp_path)) == "Nothing to undo."


def test_second_undo_is_noop(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    run_user_turn(
        model=ScriptedModel(_write_then_answer("hello.txt", "hello\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="create hello",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert undo_last_writes(session=session, store=store, cwd=str(tmp_path)) == "Nothing to undo."


def test_undo_refuses_escaped_path(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    session.checkpoint = [{"path": "../secret.txt", "existed": True, "content": "stolen"}]
    report = undo_last_writes(session=session, cwd=str(tmp_path))
    assert "skipped" in report
    assert not (tmp_path.parent / "secret.txt").exists()


def test_repl_undo_command(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    lines = iter(["make file", "/undo", "/exit"])
    model = ScriptedModel(_write_then_answer("hello.txt", "hello\n"))

    run_repl(
        model=model,
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    assert not (tmp_path / "hello.txt").exists()
    loaded = store.load(session.id)
    assert loaded.messages[-1]["content"] == "done"
    assert loaded.checkpoint == []


def test_checkpoint_persists_across_reload(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    run_user_turn(
        model=ScriptedModel(_write_then_answer("hello.txt", "hello\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="create hello",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    reloaded = store.load(session.id)
    undo_last_writes(session=reloaded, store=store, cwd=str(tmp_path))
    assert not (tmp_path / "hello.txt").exists()


def test_same_file_aliased_paths_share_snapshot(tmp_path: Path) -> None:
    checkpoint = WriteCheckpoint()
    target = tmp_path / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    checkpoint.record_target(str(tmp_path), target, existed=True, content="old\n")
    checkpoint.record("notes.txt", existed=True, content="should-not-win")
    assert checkpoint.to_list() == [{"path": "notes.txt", "existed": True, "content": "old\n"}]
