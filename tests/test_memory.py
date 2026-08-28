from pathlib import Path

from microcode.memory import ProjectMemory, apply_system_message, parse_memory_command
from microcode.repl import run_repl
from microcode.session import SessionStore
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.seen.append(list(messages))
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
                run=lambda input_data, _context: ToolResult(ok=True, output="ok"),
            )
        ]
    )


def test_read_missing_memory_is_empty(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    assert memory.read() == ""
    assert "No project memory yet" in memory.display()


def test_inject_appends_memory_section(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    injected = memory.inject("You are MicroCode.")
    assert injected.startswith("You are MicroCode.")
    assert "## Project memory" in injected
    assert "Use pytest" in injected


def test_inject_empty_leaves_system_prompt(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    assert memory.inject("You are MicroCode.") == "You are MicroCode."


def test_inject_truncates_long_memory(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, max_chars=20)
    memory.append("x" * 100)
    injected = memory.inject("sys")
    assert "memory truncated" in injected
    assert len(memory.read()) > 20


def test_append_creates_markdown_file(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    report = memory.append("Prefer read_file over guessing")
    assert report.startswith("Remembered:")
    text = (tmp_path / ".microcode" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- Prefer read_file over guessing" in text


def test_apply_system_message_replaces_leading_system() -> None:
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    updated = apply_system_message(messages, "new")
    assert updated[0] == {"role": "system", "content": "new"}
    assert updated[1]["content"] == "hi"


def test_parse_memory_command() -> None:
    assert parse_memory_command("/memory") == ("show", "")
    assert parse_memory_command("/memory add Use pytest") == ("add", "Use pytest")
    assert parse_memory_command("/memory add") == ("add", "")
    assert parse_memory_command("/memory weird") == ("help", "")


def test_repl_memory_add_refreshes_system(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    memory = ProjectMemory(tmp_path)
    base = "You are MicroCode."
    session = store.create(str(tmp_path), [{"role": "system", "content": base}])
    store.save(session)
    lines = iter(["/memory add Use pytest", "hello", "/exit"])
    model = ScriptedModel([AgentStep(type="assistant", content="ok")])

    run_repl(
        model=model,
        tools=_echo_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        memory=memory,
        system_prompt=base,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    loaded = store.load(session.id)
    assert "Use pytest" in str(loaded.messages[0]["content"])
    assert loaded.messages[1]["content"] == "hello"
    assert "Use pytest" in str(model.seen[0][0]["content"])
    assert (tmp_path / ".microcode" / "MEMORY.md").is_file()
