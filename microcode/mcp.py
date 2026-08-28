from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from microcode.tooling import ToolDefinition, ToolResult

MCP_CONFIG_NAME = "mcp.json"
JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2024-11-05"


@dataclass
class McpServerStatus:
    name: str
    command: str
    status: str
    tool_count: int = 0
    error: str | None = None
    protocol: str | None = None


@dataclass
class McpBundle:
    tools: list[ToolDefinition] = field(default_factory=list)
    statuses: list[McpServerStatus] = field(default_factory=list)
    _clients: list[StdioMcpClient] = field(default_factory=list)

    def close(self) -> None:
        for client in self._clients:
            client.close()
        self._clients.clear()

    def format_status(self) -> str:
        if not self.statuses:
            return "No MCP servers. Add .microcode/mcp.json"
        lines = []
        for item in self.statuses:
            extra = f"  {item.error}" if item.error else ""
            lines.append(
                f"{item.name}  {item.status}  tools={item.tool_count}  "
                f"protocol={item.protocol or '-'}{extra}"
            )
        return "\n".join(lines)


class StdioMcpClient:
    """One MCP server process speaking JSON-RPC over stdin/stdout."""

    def __init__(self, server_name: str, config: dict[str, Any], cwd: str) -> None:
        self.server_name = server_name
        self.config = config
        self.cwd = cwd
        self.process: subprocess.Popen[bytes] | None = None
        self.protocol: str | None = None
        self._next_id = 1
        self._pending: dict[int, Queue[Any]] = {}
        self._lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.stderr_lines: list[str] = []

    def start(self) -> None:
        last_error: Exception | None = None
        for protocol in self._protocol_candidates():
            try:
                self._spawn()
                self.protocol = protocol
                self._ensure_stdout_thread()
                self.request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "microcode", "version": "0.1.0"},
                    },
                    timeout_seconds=15.0,
                )
                self.notify("notifications/initialized", {})
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                self.close()
        raise RuntimeError(str(last_error or f'MCP "{self.server_name}" failed to start'))

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list", {}, timeout_seconds=15.0)
        if not isinstance(result, dict):
            return []
        tools = result.get("tools") or []
        return [item for item in tools if isinstance(item, dict)]

    def call_tool(self, name: str, arguments: Any) -> ToolResult:
        payload = arguments if isinstance(arguments, dict) else {}
        result = self.request(
            "tools/call",
            {"name": name, "arguments": payload},
            timeout_seconds=60.0,
        )
        return _format_tool_result(result)

    def request(self, method: str, params: Any, timeout_seconds: float = 15.0) -> Any:
        message_id = self._next_id
        self._next_id += 1
        waiter: Queue[Any] = Queue(maxsize=1)
        with self._lock:
            self._pending[message_id] = waiter
        self.send({"jsonrpc": JSONRPC_VERSION, "id": message_id, "method": method, "params": params})
        try:
            message = waiter.get(timeout=timeout_seconds)
        except Empty as error:
            with self._lock:
                self._pending.pop(message_id, None)
            extra = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f'MCP {self.server_name}: timeout on {method}' + (f"\n{extra}" if extra else "")
            ) from error
        if message.get("error"):
            err = message["error"]
            text = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"MCP {self.server_name}: {text}")
        return message.get("result")

    def notify(self, method: str, params: Any) -> None:
        self.send({"jsonrpc": JSONRPC_VERSION, "method": method, "params": params})

    def send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f'MCP "{self.server_name}" is not running')
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        if self.protocol == "newline-json":
            self.process.stdin.write(payload + b"\n")
        else:
            header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
            self.process.stdin.write(header + payload)
        self.process.stdin.flush()

    def close(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for queue in pending:
            queue.put({"error": {"message": f'MCP "{self.server_name}" closed'}})
        process = self.process
        self.process = None
        self.protocol = None
        self._stdout_thread = None
        self._stderr_thread = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                process.kill()
            except OSError:
                pass

    def _protocol_candidates(self) -> list[str]:
        configured = str(self.config.get("protocol") or "").strip().lower()
        if configured in {"newline-json", "content-length"}:
            return [configured]
        return ["content-length", "newline-json"]

    def _spawn(self) -> None:
        command = str(self.config.get("command") or "").strip()
        if not command:
            raise RuntimeError(f'MCP "{self.server_name}" has no command')
        args = [str(item) for item in (self.config.get("args") or [])]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for key, value in dict(self.config.get("env") or {}).items():
            env[str(key)] = str(value)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        self.process = subprocess.Popen(  # noqa: S603
            [command, *args],
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **kwargs,
        )
        self.stderr_lines = []
        with self._lock:
            self._pending = {}
        self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()

    def _ensure_stdout_thread(self) -> None:
        if self._stdout_thread is not None:
            return
        self._stdout_thread = threading.Thread(target=self._consume_stdout, daemon=True)
        self._stdout_thread.start()

    def _consume_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        for raw in self.process.stderr:
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                self.stderr_lines = (self.stderr_lines + [text])[-12:]

    def _consume_stdout(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        stdout = self.process.stdout
        try:
            while True:
                line = stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                stripped = text.strip()
                if not stripped:
                    continue
                if self.protocol == "newline-json" or (
                    self.protocol != "content-length" and not text.lower().startswith("content-length:")
                ):
                    try:
                        self._handle_message(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                    continue
                headers = [text.rstrip("\r\n")]
                while True:
                    next_line = stdout.readline()
                    if not next_line:
                        return
                    header = next_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if header == "":
                        break
                    headers.append(header)
                length = 0
                for header in headers:
                    if header.lower().startswith("content-length:"):
                        try:
                            length = int(header.split(":", 1)[1].strip())
                        except ValueError:
                            length = 0
                if length <= 0:
                    continue
                body = stdout.read(length)
                try:
                    self._handle_message(json.loads(body.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        finally:
            with self._lock:
                waiters = list(self._pending.values())
                self._pending.clear()
            for queue in waiters:
                queue.put({"error": {"message": f'MCP "{self.server_name}" exited'}})

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if not isinstance(message_id, int):
            return
        with self._lock:
            queue = self._pending.pop(message_id, None)
        if queue is not None:
            queue.put(message)


def load_mcp_servers(cwd: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(cwd) / ".microcode" / MCP_CONFIG_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return {}
    return {
        str(name): value
        for name, value in servers.items()
        if isinstance(value, dict)
    }


def connect_mcp(cwd: str, servers: dict[str, dict[str, Any]] | None = None) -> McpBundle:
    configured = servers if servers is not None else load_mcp_servers(cwd)
    bundle = McpBundle()
    for name, config in configured.items():
        if config.get("enabled") is False:
            bundle.statuses.append(
                McpServerStatus(name=name, command=str(config.get("command") or ""), status="disabled")
            )
            continue
        client = StdioMcpClient(name, config, cwd)
        try:
            client.start()
            descriptors = client.list_tools()
            bundle._clients.append(client)
            for descriptor in descriptors:
                bundle.tools.append(_wrap_tool(client, name, descriptor))
            bundle.statuses.append(
                McpServerStatus(
                    name=name,
                    command=str(config.get("command") or ""),
                    status="connected",
                    tool_count=len(descriptors),
                    protocol=client.protocol,
                )
            )
        except Exception as error:  # noqa: BLE001
            client.close()
            bundle.statuses.append(
                McpServerStatus(
                    name=name,
                    command=str(config.get("command") or ""),
                    status="error",
                    error=str(error),
                )
            )
    return bundle


def _wrap_tool(client: StdioMcpClient, server_name: str, descriptor: dict[str, Any]) -> ToolDefinition:
    remote_name = str(descriptor.get("name") or "tool")
    wrapped = f"mcp__{_safe_segment(server_name)}__{_safe_segment(remote_name)}"
    schema = descriptor.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "additionalProperties": True}

    def _run(input_data: Any, _context: Any, *, _client=client, _name=remote_name) -> ToolResult:
        try:
            return _client.call_tool(_name, input_data)
        except Exception as error:  # noqa: BLE001
            return ToolResult(ok=False, output=str(error))

    return ToolDefinition(
        name=wrapped,
        description=str(descriptor.get("description") or f"MCP tool {remote_name} from {server_name}"),
        input_schema=schema,
        validator=lambda value: value if isinstance(value, dict) else {},
        run=_run,
    )


def _safe_segment(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    cleaned = "".join(chars).strip("_")
    return cleaned or "tool"


def _format_tool_result(result: Any) -> ToolResult:
    if not isinstance(result, dict):
        return ToolResult(ok=True, output=json.dumps(result, ensure_ascii=False))
    parts: list[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
    if not parts:
        parts.append(json.dumps(result, ensure_ascii=False))
    return ToolResult(ok=not bool(result.get("isError")), output="\n".join(part for part in parts if part).strip())
