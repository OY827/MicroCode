from pathlib import Path
from unittest.mock import MagicMock, patch

from microcode.mock_model import MockModelAdapter
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.web_search import parse_search_results

SAMPLE_HTML = """
<html><body>
<a class="result__a" href="/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fpathlib.html">
  Pathlib <b>docs</b>
</a>
<a class="result__snippet">Working with filesystem paths.</a>
<a class="result__a" href="https://example.com/second">Second hit</a>
<div class="result__snippet">Another snippet &amp; more</div>
</body></html>
"""


def test_parse_search_results_unwraps_uddg_and_strips_tags() -> None:
    results = parse_search_results(SAMPLE_HTML, 5)
    assert len(results) == 2
    assert results[0]["title"] == "Pathlib docs"
    assert results[0]["url"] == "https://docs.python.org/3/library/pathlib.html"
    assert "filesystem paths" in results[0]["snippet"]
    assert results[1]["url"] == "https://example.com/second"
    assert results[1]["snippet"] == "Another snippet & more"


def test_parse_search_results_respects_limit() -> None:
    assert len(parse_search_results(SAMPLE_HTML, 1)) == 1


def test_web_search_can_be_rejected(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "web_search",
        {"query": "pathlib Path"},
        ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
    )
    assert not result.ok
    assert "rejected" in result.output


def test_web_search_formats_mocked_page(tmp_path: Path) -> None:
    response = MagicMock()
    response.read.return_value = SAMPLE_HTML.encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=response) as opener:
        result = create_default_tool_registry().execute(
            "web_search",
            {"query": "pathlib", "num_results": 5},
            ToolContext(cwd=str(tmp_path)),
        )

    assert result.ok
    assert "Search results for: pathlib" in result.output
    assert "https://docs.python.org/3/library/pathlib.html" in result.output
    assert "Pathlib docs" in result.output
    opener.assert_called_once()
    request = opener.call_args[0][0]
    assert "html.duckduckgo.com" in request.full_url
    assert "pathlib" in request.full_url


def test_web_search_empty_page_is_error(tmp_path: Path) -> None:
    response = MagicMock()
    response.read.return_value = b"<html></html>"
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=response):
        result = create_default_tool_registry().execute(
            "web_search",
            {"query": "zzzz-no-hits"},
            ToolContext(cwd=str(tmp_path)),
        )

    assert not result.ok
    assert "No search results" in result.output


def test_web_search_rejects_bad_count(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "web_search",
        {"query": "x", "num_results": 0},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "num_results" in result.output


def test_mock_model_search() -> None:
    step = MockModelAdapter().next([{"role": "user", "content": "/search pathlib Path"}])
    assert step.type == "tool_calls"
    assert step.calls[0]["toolName"] == "web_search"
    assert step.calls[0]["input"] == {"query": "pathlib Path"}
