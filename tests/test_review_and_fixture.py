import importlib.util

from microcode.mock_model import MockModelAdapter
from microcode.repl import apply_local_command
from microcode.session import SessionStore
from microcode.tooling import ToolContext, ToolRegistry, WriteDecision
from microcode.tools import create_default_tool_registry
from microcode.turn_tape import TurnTape


def test_revise_write_saves_edited_text(tmp_path) -> None:
    tools = create_default_tool_registry()

    def revise(_path: str, _proposed: str, _preview: str) -> WriteDecision:
        return WriteDecision(allow=True, content="edited\n")

    result = tools.execute(
        "write_file",
        {"path": "notes.txt", "content": "from the model\n"},
        ToolContext(cwd=str(tmp_path), on_revise_write=revise),
    )
    assert result.ok
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "edited\n"
    assert "edited before write" in result.output


def test_revise_write_reject_leaves_disk_unchanged(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    tools = create_default_tool_registry()
    result = tools.execute(
        "write_file",
        {"path": "notes.txt", "content": "new\n"},
        ToolContext(
            cwd=str(tmp_path),
            on_revise_write=lambda *_args: WriteDecision(allow=False),
        ),
    )
    assert not result.ok
    assert target.read_text(encoding="utf-8") == "old\n"


def test_fixture_groups_calls_from_tape_and_replays(tmp_path) -> None:
    store = SessionStore(tmp_path)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look"},
        {
            "role": "assistant_tool_call",
            "toolUseId": "1",
            "toolName": "read_file",
            "input": {"path": "a.py"},
        },
        {
            "role": "tool_result",
            "toolUseId": "1",
            "toolName": "read_file",
            "content": "print(1)\n",
            "isError": False,
        },
        {
            "role": "assistant_tool_call",
            "toolUseId": "2",
            "toolName": "grep_files",
            "input": {"pattern": "print"},
        },
        {
            "role": "tool_result",
            "toolUseId": "2",
            "toolName": "grep_files",
            "content": "a.py:1\n",
            "isError": False,
        },
        {"role": "assistant", "content": "found the print"},
    ]
    session = store.create(str(tmp_path), messages)
    tape = TurnTape(tmp_path, session_id=session.id)
    tape.record_user("look")
    from microcode.types import AgentStep

    tape.record_model(
        model="mock",
        step=AgentStep(
            type="tool_calls",
            calls=[
                {"id": "1", "toolName": "read_file", "input": {"path": "a.py"}},
                {"id": "2", "toolName": "grep_files", "input": {"pattern": "print"}},
            ],
        ),
        roles=["system", "user"],
    )
    tape.record_model(
        model="mock",
        step=AgentStep(type="assistant", content="found the print"),
        roles=["system", "user"],
    )
    result = apply_local_command(
        "/fixture",
        model=MockModelAdapter(),
        tools=ToolRegistry([]),
        messages=messages,
        cwd=str(tmp_path),
        session=session,
    )
    assert result.handled
    path = tmp_path / ".microcode" / "fixtures" / f"{session.id}.py"
    assert path.is_file()
    assert str(path) in result.output
    source = path.read_text(encoding="utf-8")
    assert source.count('"type": "tool_calls"') == 1 or "tool_calls" in source

    spec = importlib.util.spec_from_file_location("replay_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.TURNS[0]["steps"][0]["calls"]) == 2
    module.test_replay_session()
