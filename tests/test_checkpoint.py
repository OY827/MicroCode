from pathlib import Path

from microcode.checkpoint import (
    WriteCheckpoint,
    format_checkpoints,
    preview_rewind,
    redo_writes,
    rewind_writes,
    undo_last_writes,
)
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


def _two_write_turns(tmp_path: Path):
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
    return session, store


def test_checkpoints_are_named_per_write_turn(tmp_path: Path) -> None:
    session, store = _two_write_turns(tmp_path)
    listed = format_checkpoints(session)
    assert "cp_1" in listed
    assert "a.txt" in listed
    assert "cp_2" in listed
    assert "b.txt" in listed
    loaded = store.load(session.id)
    assert [item["id"] for item in loaded.checkpoints] == ["cp_1", "cp_2"]


def test_rewind_last_group_keeps_earlier_writes(tmp_path: Path) -> None:
    session, store = _two_write_turns(tmp_path)
    preview = preview_rewind(session=session)
    assert "cp_2" in preview
    assert "delete b.txt" not in preview
    assert "restore b.txt" in preview
    report = rewind_writes(session=session, store=store, cwd=str(tmp_path))
    assert "restored b.txt" in report
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A1\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B0\n"
    assert [item["id"] for item in session.checkpoints] == ["cp_1"]


def test_rewind_to_first_checkpoint_restores_all_later_writes(tmp_path: Path) -> None:
    session, store = _two_write_turns(tmp_path)
    preview = preview_rewind(session=session, target_id="cp_1")
    assert "2 checkpoint(s)" in preview
    assert "restore a.txt" in preview
    assert "restore b.txt" in preview
    rewind_writes(session=session, store=store, cwd=str(tmp_path), target_id="cp_1")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A0\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B0\n"
    assert session.checkpoints == []
    loaded = store.load(session.id)
    assert loaded.checkpoints == []


def test_rewind_unknown_id(tmp_path: Path) -> None:
    session, _store = _two_write_turns(tmp_path)
    assert preview_rewind(session=session, target_id="cp_99") == "Checkpoint not found: cp_99"
    assert rewind_writes(session=session, cwd=str(tmp_path), target_id="cp_99") == (
        "Checkpoint not found: cp_99"
    )


def test_undo_pops_last_checkpoint_from_history(tmp_path: Path) -> None:
    session, store = _two_write_turns(tmp_path)
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert [item["id"] for item in session.checkpoints] == ["cp_1"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B0\n"


def test_repl_rewind_command(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A0\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    lines = iter(["edit a", "/checkpoints", "/rewind-preview", "/rewind", "/exit"])
    model = ScriptedModel(_write_then_answer("a.txt", "A1\n"))

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

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A0\n"
    loaded = store.load(session.id)
    assert loaded.checkpoints == []


def test_restore_deletes_created_directory(tmp_path: Path) -> None:
    dest = tmp_path / "copied"
    dest.mkdir()
    (dest / "a.txt").write_text("x\n", encoding="utf-8")
    checkpoint = WriteCheckpoint()
    checkpoint.record("copied", existed=False, is_dir=True)
    report = "\n".join(checkpoint.restore(str(tmp_path)))
    assert "deleted copied" in report
    assert not dest.exists()


def test_restore_recreates_empty_directory(tmp_path: Path) -> None:
    checkpoint = WriteCheckpoint()
    checkpoint.record("empty", existed=True, is_dir=True)
    checkpoint.restore(str(tmp_path))
    assert (tmp_path / "empty").is_dir()


def test_undo_then_redo_restores_new_content(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    run_user_turn(
        model=ScriptedModel(_write_then_answer("notes.txt", "new\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        user_text="edit notes",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"
    assert session.checkpoints == []
    listed = format_checkpoints(session)
    assert "can redo: 1 step(s)" in listed
    report = redo_writes(session=session, store=store, cwd=str(tmp_path))
    assert "restored notes.txt" in report
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\n"
    assert [item["id"] for item in session.checkpoints] == ["cp_1"]
    assert session.undone == []
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"


def test_rewind_then_redo_restores_later_writes(tmp_path: Path) -> None:
    session, store = _two_write_turns(tmp_path)
    rewind_writes(session=session, store=store, cwd=str(tmp_path), target_id="cp_1")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A0\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B0\n"
    assert session.checkpoints == []
    redo_writes(session=session, store=store, cwd=str(tmp_path))
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A1\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B1\n"
    assert [item["id"] for item in session.checkpoints] == ["cp_1", "cp_2"]


def test_redo_persists_across_reload(tmp_path: Path) -> None:
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
    assert not (tmp_path / "hello.txt").exists()
    reloaded = store.load(session.id)
    assert len(reloaded.undone) == 1
    redo_writes(session=reloaded, store=store, cwd=str(tmp_path))
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\n"


def test_new_write_clears_redo_stack(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    tools = create_default_tool_registry()
    run_user_turn(
        model=ScriptedModel(_write_then_answer("a.txt", "A\n")),
        tools=tools,
        messages=session.messages,
        user_text="create a",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    undo_last_writes(session=session, store=store, cwd=str(tmp_path))
    assert session.undone
    run_user_turn(
        model=ScriptedModel(_write_then_answer("b.txt", "B\n")),
        tools=tools,
        messages=session.messages,
        user_text="create b",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )
    assert session.undone == []
    assert redo_writes(session=session, store=store, cwd=str(tmp_path)) == "Nothing to redo."
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "B\n"


def test_repl_redo_command(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    lines = iter(["make file", "/undo", "/redo", "/exit"])
    run_repl(
        model=ScriptedModel(_write_then_answer("hello.txt", "hello\n")),
        tools=create_default_tool_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    loaded = store.load(session.id)
    assert [item["id"] for item in loaded.checkpoints] == ["cp_1"]
    assert loaded.undone == []
