from pathlib import Path

from microcode.prompt import build_system_prompt
from microcode.repl import run_repl
from microcode.skills import discover_skills, format_skill_catalog, load_skill
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
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


def _write_skill(root: Path, name: str, body: str) -> Path:
    skill_dir = root / ".microcode" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_discover_and_load_skill(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "pytest",
        "# Pytest\n\nRun tests with pytest -q.\n\nAlways use the project venv.\n",
    )
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "pytest"
    assert "pytest -q" in skills[0].description

    loaded = load_skill(str(tmp_path), "pytest")
    assert loaded is not None
    assert "Always use the project venv" in loaded.content


def test_unknown_and_unsafe_names(tmp_path: Path) -> None:
    assert load_skill(str(tmp_path), "missing") is None
    assert load_skill(str(tmp_path), "../sessions") is None
    assert load_skill(str(tmp_path), "foo/bar") is None


def test_load_skill_tool_returns_markdown(tmp_path: Path) -> None:
    _write_skill(tmp_path, "commit", "# Commit\n\nWrite conventional commits.\n")
    tools = create_default_tool_registry()
    result = tools.execute(
        "load_skill",
        {"name": "commit"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "SKILL: commit" in result.output
    assert "Write conventional commits" in result.output

    missing = tools.execute("load_skill", {"name": "nope"}, ToolContext(cwd=str(tmp_path)))
    assert not missing.ok
    assert "Unknown skill" in missing.output


def test_skill_catalog_in_system_prompt(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pytest", "# Pytest\n\nRun tests with pytest -q.\n")
    prompt = build_system_prompt("You are MicroCode.", str(tmp_path))
    catalog = format_skill_catalog(discover_skills(tmp_path))
    assert "## Local skills" in prompt
    assert "load_skill" in catalog
    assert "pytest" in prompt


def test_repl_skills_command(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pytest", "# Pytest\n\nRun tests with pytest -q.\n")
    lines = iter(["/skills", "/exit"])
    run_repl(
        model=ScriptedModel([]),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
        read_line=lambda: next(lines) + "\n",
    )
