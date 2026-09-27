from pathlib import Path

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.test_runner import (
    build_test_command,
    discover_test_files,
    failed_test_names,
    parse_pytest_summary,
)


def test_discover_test_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "tests" / "util_test.py").write_text("def test_util():\n    assert True\n", encoding="utf-8")
    (tmp_path / ".venv" / "test_ignore.py").parent.mkdir()
    (tmp_path / ".venv" / "test_ignore.py").write_text("def test_no():\n    assert False\n", encoding="utf-8")
    found = {path.name for path in discover_test_files(tmp_path)}
    assert found == {"test_app.py", "util_test.py"}


def test_parse_pytest_summary_and_failures() -> None:
    output = (
        "tests/test_app.py::test_ok PASSED\n"
        "FAILED tests/test_app.py::test_bad - assert 1 == 2\n"
        "===== 1 passed, 1 failed, 1 skipped in 0.10s =====\n"
    )
    counts = parse_pytest_summary(output)
    assert counts == {"passed": 1, "failed": 1, "errors": 0, "skipped": 1}
    assert failed_test_names(output) == ["tests/test_app.py::test_bad - assert 1 == 2"]


def test_build_pytest_command_includes_pattern(tmp_path: Path) -> None:
    command = build_test_command("pytest", tmp_path / "tests", pattern="login")
    assert command[1:4] == ["-m", "pytest", str(tmp_path / "tests")]
    assert "-k" in command
    assert "login" in command


def test_test_runner_runs_passing_tests(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert 1 == 1\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "test_runner",
        {"path": "tests", "framework": "pytest"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "passed: 1" in result.output
    assert "failed: 0" in result.output
    assert "test_ok.py" in result.output


def test_test_runner_reports_failures(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "test_runner",
        {"path": "tests", "framework": "pytest"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "failed: 1" in result.output
    assert "test_bad" in result.output


def test_test_runner_no_tests(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "test_runner",
        {"path": "src"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "No test files" in result.output


def test_test_runner_can_be_rejected(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "test_runner",
        {"path": "tests", "framework": "pytest"},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output


def test_mock_model_test_runner() -> None:
    step = MockModelAdapter().next([{"role": "user", "content": "/test tests"}])
    assert step.type == "tool_calls"
    assert step.calls[0]["toolName"] == "test_runner"
    assert step.calls[0]["input"] == {"path": "tests"}
