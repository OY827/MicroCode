from pathlib import Path

from microcode.memory import ProjectMemory
from microcode.project_init import INIT_RELATIVE, load_init_markdown, run_init, scan_project
from microcode.prompt import build_system_prompt
from microcode.repl import handle_local_command, run_repl
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.calls += 1
        return AgentStep(type="assistant", content="model ran")


def test_scan_detects_pytest_and_env_rules(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    report = scan_project(tmp_path)
    assert any("pytest" in item.lower() for item in report.how_to_test)
    assert ".env" in report.leave_alone
    assert ".venv" in report.leave_alone


def test_run_init_writes_file_and_memory(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    memory = ProjectMemory(tmp_path)
    text = run_init(tmp_path, memory)
    path = tmp_path / INIT_RELATIVE
    assert path.is_file()
    assert "How to test" in text
    assert "Do not touch" in path.read_text(encoding="utf-8")
    notes = memory.manager.all_entries()
    assert len(notes) == 1
    assert "Project init" in notes[0].content
    again = run_init(tmp_path, memory)
    assert "Wrote" in again
    assert len(memory.manager.all_entries()) == 1


def test_system_prompt_includes_init_file(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    run_init(tmp_path)
    prompt = build_system_prompt("You are MicroCode.", str(tmp_path))
    assert "Workspace init" in prompt
    assert "How to test" in prompt
    assert load_init_markdown(tmp_path)


def test_repl_init_does_not_call_the_model(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    memory = ProjectMemory(tmp_path)
    model = CountingModel()
    lines = iter(["/init", "/exit"])
    run_repl(
        model=model,
        tools=ToolRegistry([]),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
        memory=memory,
        system_prompt="You are MicroCode.",
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    assert model.calls == 0
    assert (tmp_path / INIT_RELATIVE).is_file()
    assert handle_local_command("/init") == "init"
