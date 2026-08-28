import json
import sys
from pathlib import Path

from microcode.mcp import connect_mcp, load_mcp_servers
from microcode.repl import handle_local_command
from microcode.tooling import ToolContext


FAKE_SERVER = Path(__file__).resolve().parent / "fake_mcp_server.py"


def test_load_mcp_servers_missing(tmp_path: Path) -> None:
    assert load_mcp_servers(tmp_path) == {}


def test_load_mcp_servers_from_json(tmp_path: Path) -> None:
    config_dir = tmp_path / ".microcode"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "command": sys.executable,
                        "args": ["-u", str(FAKE_SERVER)],
                        "protocol": "newline-json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    servers = load_mcp_servers(tmp_path)
    assert "fake" in servers
    assert servers["fake"]["protocol"] == "newline-json"


def test_connect_mcp_echo_tool(tmp_path: Path) -> None:
    bundle = connect_mcp(
        str(tmp_path),
        {
            "fake": {
                "command": sys.executable,
                "args": ["-u", str(FAKE_SERVER)],
                "protocol": "newline-json",
            }
        },
    )
    try:
        assert bundle.statuses[0].status == "connected"
        assert bundle.tools
        tool = bundle.tools[0]
        assert tool.name == "mcp__fake__echo"
        result = tool.run({"text": "hi"}, ToolContext(cwd=str(tmp_path)))
        assert result.ok
        assert "echo:hi" in result.output
    finally:
        bundle.close()


def test_connect_mcp_bad_command_is_error(tmp_path: Path) -> None:
    bundle = connect_mcp(
        str(tmp_path),
        {"missing": {"command": "microcode-mcp-does-not-exist-xyz", "protocol": "newline-json"}},
    )
    try:
        assert bundle.tools == []
        assert bundle.statuses[0].status == "error"
        assert "mcp:\n" + bundle.format_status()
    finally:
        bundle.close()


def test_handle_local_mcp() -> None:
    assert handle_local_command("/mcp") == "mcp"
