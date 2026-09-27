from pathlib import Path

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.code_review import review_source


def test_review_unused_import_uses_ast_not_substring() -> None:
    source = "import os\n\ntext = 'os'\nprint(text)\n"
    kinds = {issue.kind for issue in review_source(source)}
    assert "unused_import" in kinds

    source_used = "import os\n\nprint(os.getcwd())\n"
    kinds_used = {issue.kind for issue in review_source(source_used)}
    assert "unused_import" not in kinds_used


def test_review_skips_future_imports() -> None:
    source = "from __future__ import annotations\n\nx = 1\n"
    assert review_source(source) == []


def test_review_bare_and_empty_except() -> None:
    source = (
        "def run():\n"
        "    try:\n"
        "        1 / 0\n"
        "    except:\n"
        "        pass\n"
    )
    kinds = {issue.kind for issue in review_source(source)}
    assert "bare_except" in kinds
    assert "empty_except" in kinds


def test_review_long_function() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(55))
    source = f"def bulky():\n{body}\n    return 1\n"
    issues = [item for item in review_source(source) if item.kind == "long_function"]
    assert issues
    assert issues[0].line == 1


def test_code_review_tool_reports_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("import json\n\nx = 1\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "code_review",
        {"path": "app.py"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "issues: 1" in result.output
    assert "unused_import" in result.output
    assert "app.py:" in result.output


def test_code_review_clean_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "code_review",
        {"path": "app.py"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "No issues found" in result.output


def test_code_review_checks_filter(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "import json\n\n"
        "def run():\n"
        "    try:\n"
        "        1 / 0\n"
        "    except:\n"
        "        pass\n",
        encoding="utf-8",
    )
    tools = create_default_tool_registry()
    imports = tools.execute("code_review", {"path": "app.py", "checks": "imports"}, ToolContext(cwd=str(tmp_path)))
    excepts = tools.execute("code_review", {"path": "app.py", "checks": "excepts"}, ToolContext(cwd=str(tmp_path)))
    assert imports.ok and "unused_import" in imports.output
    assert "bare_except" not in imports.output
    assert excepts.ok and "bare_except" in excepts.output
    assert "unused_import" not in excepts.output


def test_code_review_rejects_non_python_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("import os\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "code_review",
        {"path": "notes.txt"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "Not a Python file" in result.output


def test_code_review_missing_path(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "code_review",
        {"path": "missing.py"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "does not exist" in result.output


def test_mock_review() -> None:
    step = MockModelAdapter().next([{"role": "user", "content": "/review src"}])
    assert step.calls[0]["toolName"] == "code_review"
    assert step.calls[0]["input"] == {"path": "src"}
