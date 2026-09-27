from microcode.command_guard import refuse_command, split_command_line
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry


def test_split_keeps_quoted_python_c() -> None:
    tokens = split_command_line('python -c "print(123)"')
    assert tokens[0].lower().startswith("python")
    assert "-c" in tokens
    assert "print(123)" in tokens


def test_refuse_pipe_and_ampersand() -> None:
    assert refuse_command("pytest | tee out.txt")
    assert refuse_command("pytest && rm -rf /")
    assert "one program" in (refuse_command("echo hi && echo bye") or "")


def test_refuse_destructive_git_and_rm() -> None:
    assert "rm -rf" in (refuse_command("rm -rf /") or "")
    assert "reset --hard" in (refuse_command("git reset --hard") or "")
    assert "push --force" in (refuse_command("git push --force origin main") or "")
    assert "git clean" in (refuse_command("git clean -fd") or "")


def test_allow_pytest_and_python() -> None:
    assert refuse_command("pytest -q") is None
    assert refuse_command('python -c "print(1 && 2)"') is None


def test_run_command_blocks_before_ask(tmp_path) -> None:
    asked: list[str] = []
    result = create_default_tool_registry().execute(
        "run_command",
        {"command": "rm -rf /"},
        ToolContext(cwd=str(tmp_path), on_approve=lambda summary: asked.append(summary) or True),
    )
    assert not result.ok
    assert "Refusing" in result.output
    assert asked == []
