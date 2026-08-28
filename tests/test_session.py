from pathlib import Path

from microcode.repl import run_user_turn
from microcode.session import SessionStore, summarize_session
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=lambda input_data, _context: ToolResult(ok=True, output=input_data["text"]),
            )
        ]
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)

    loaded = store.load()
    assert loaded.id == session.id
    assert loaded.messages[0]["content"] == "sys"
    assert (tmp_path / ".microcode" / "sessions" / f"{session.id}.json").is_file()


def test_load_missing_session_raises(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    try:
        store.load("missing")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_user_turn_persists_messages(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    store.save(session)
    model = ScriptedModel([AgentStep(type="assistant", content="done")])

    run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=session.messages,
        user_text="hello",
        cwd=str(tmp_path),
        session=session,
        store=store,
    )

    loaded = store.load(session.id)
    assert [message["role"] for message in loaded.messages] == ["system", "user", "assistant"]
    assert loaded.messages[1]["content"] == "hello"
    assert loaded.messages[-1]["content"] == "done"
    assert "id:" in summarize_session(loaded)
