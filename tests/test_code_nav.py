from pathlib import Path

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.code_nav import extract_name_uses, extract_symbols

SAMPLE = '''
# def login(): pass
def login(user):
    return user.name

class Auth:
    def check(self):
        login(self.user)

message = "please login"
login_count = 1
'''


def test_extract_symbols_finds_function_and_class() -> None:
    hits = extract_symbols(SAMPLE)
    names = {(kind, name) for kind, name, _line, _extra in hits}
    assert ("function", "login") in names
    assert ("function", "check") in names
    assert ("class", "Auth") in names
    login = next(item for item in hits if item[1] == "login")
    assert login[3] == "(user)"


def test_extract_name_uses_skips_comments_and_strings() -> None:
    uses = extract_name_uses(SAMPLE, "login")
    kinds = {kind for kind, _line in uses}
    assert "definition" in kinds
    assert "use" in kinds
    source_lines = SAMPLE.splitlines()
    matched = {source_lines[line - 1].strip() for _kind, line in uses}
    assert any(line.startswith("def login") for line in matched)
    assert any("login(self.user)" in line for line in matched)
    assert not any(line.startswith("#") for line in matched)
    assert "message = \"please login\"" not in matched
    assert "login_count = 1" not in matched


def test_find_symbols_tool(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def login(user):\n    return user\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "find_symbols",
        {"name": "login"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "function  login(user)" in result.output
    assert "auth.py:1" in result.output


def test_find_symbols_skips_comment_only_match(tmp_path: Path) -> None:
    (tmp_path / "notes.py").write_text("# def login():\npass\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "find_symbols",
        {"name": "login"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "No login found" in result.output


def test_find_references_tool(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "def login():\n    pass\n\nlogin()\n",
        encoding="utf-8",
    )
    result = create_default_tool_registry().execute(
        "find_references",
        {"name": "login"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "definition" in result.output
    assert "use" in result.output
    assert "auth.py:1" in result.output
    assert "auth.py:4" in result.output


def test_find_symbols_skips_syntax_error(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("def (\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("def good():\n    pass\n", encoding="utf-8")
    result = create_default_tool_registry().execute(
        "find_symbols",
        {"symbol_type": "function"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.ok
    assert "good" in result.output
    assert "bad.py" not in result.output


def test_mock_symbols_and_refs() -> None:
    symbols = MockModelAdapter().next([{"role": "user", "content": "/symbols login"}])
    assert symbols.calls[0]["toolName"] == "find_symbols"
    refs = MockModelAdapter().next([{"role": "user", "content": "/refs login"}])
    assert refs.calls[0]["toolName"] == "find_references"
    assert refs.calls[0]["input"] == {"name": "login"}
