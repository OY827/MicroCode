from pathlib import Path
from unittest.mock import MagicMock, patch

from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.tools.web_fetch import url_is_allowed


def test_url_is_allowed_rejects_localhost() -> None:
    ok, reason = url_is_allowed("https://127.0.0.1/secret")
    assert ok is False
    assert "blocked" in reason


def test_url_is_allowed_rejects_non_http() -> None:
    ok, reason = url_is_allowed("file:///etc/passwd")
    assert ok is False
    assert "http" in reason


def test_web_fetch_blocks_loopback_without_network(tmp_path: Path) -> None:
    result = create_default_tool_registry().execute(
        "web_fetch",
        {"url": "http://127.0.0.1/"},
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.ok
    assert "blocked" in result.output


def test_web_fetch_can_be_rejected(tmp_path: Path) -> None:
    with patch("microcode.tools.web_fetch.url_is_allowed", return_value=(True, "ok")):
        result = create_default_tool_registry().execute(
            "web_fetch",
            {"url": "https://example.com/docs"},
            ToolContext(cwd=str(tmp_path), on_approve=lambda _summary: False),
        )
    assert not result.ok
    assert "rejected" in result.output


def test_web_fetch_returns_text(tmp_path: Path) -> None:
    response = MagicMock()
    response.headers = {"Content-Type": "text/plain; charset=utf-8"}
    response.status = 200
    response.read.return_value = b"hello docs"
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    opener = MagicMock()
    opener.open.return_value = response

    with (
        patch("microcode.tools.web_fetch.url_is_allowed", return_value=(True, "ok")),
        patch("urllib.request.build_opener", return_value=opener),
    ):
        result = create_default_tool_registry().execute(
            "web_fetch",
            {"url": "https://example.com/docs"},
            ToolContext(cwd=str(tmp_path)),
        )

    assert result.ok
    assert "hello docs" in result.output
    assert "STATUS: 200" in result.output
    opener.open.assert_called_once()


def test_web_fetch_strips_html(tmp_path: Path) -> None:
    response = MagicMock()
    response.headers = {"Content-Type": "text/html"}
    response.status = 200
    response.read.return_value = b"<html><script>x()</script><p>Hello</p></html>"
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    opener = MagicMock()
    opener.open.return_value = response

    with (
        patch("microcode.tools.web_fetch.url_is_allowed", return_value=(True, "ok")),
        patch("urllib.request.build_opener", return_value=opener),
    ):
        result = create_default_tool_registry().execute(
            "web_fetch",
            {"url": "https://example.com/page"},
            ToolContext(cwd=str(tmp_path)),
        )

    assert result.ok
    assert "Hello" in result.output
    assert "x()" not in result.output
    assert "<p>" not in result.output
