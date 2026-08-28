from io import BytesIO

from microcode.anthropic_adapter import (
    anthropic_messages_url,
    consume_anthropic_stream,
    parse_anthropic_response,
    to_anthropic_messages,
)
from microcode.api_client import iter_sse_or_json
from microcode.config import ModelConfig, resolve_protocol
from microcode.models import create_model_adapter
from microcode.openai_adapter import consume_openai_stream, openai_chat_url, parse_openai_response, to_openai_messages
from microcode.tooling import ToolRegistry


def test_create_model_adapter_selects_anthropic() -> None:
    config = ModelConfig(
        api_key="sk-test",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-v4-pro",
    )
    from microcode.anthropic_adapter import AnthropicModelAdapter

    adapter = create_model_adapter(config, ToolRegistry([]))
    assert isinstance(adapter, AnthropicModelAdapter)
    assert anthropic_messages_url(config.base_url) == (
        "https://api.deepseek.com/anthropic/v1/messages"
    )


def test_openai_url_and_protocol() -> None:
    config = ModelConfig(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    assert resolve_protocol(config) == "openai"
    assert openai_chat_url(config.base_url) == "https://api.deepseek.com/v1/chat/completions"


def test_to_anthropic_messages_groups_tool_roundtrip() -> None:
    system, converted = to_anthropic_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant_tool_call",
                "toolUseId": "1",
                "toolName": "list_files",
                "input": {},
            },
            {
                "role": "tool_result",
                "toolUseId": "1",
                "toolName": "list_files",
                "content": "file a.py",
                "isError": False,
            },
        ]
    )
    assert system == "sys"
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"


def test_parse_anthropic_tool_use() -> None:
    step = parse_anthropic_response(
        {
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "list_files",
                    "input": {"path": "."},
                },
            ]
        }
    )
    assert step.type == "tool_calls"
    assert step.calls[0]["toolName"] == "list_files"
    assert step.calls[0]["input"] == {"path": "."}


def test_parse_anthropic_text() -> None:
    step = parse_anthropic_response({"content": [{"type": "text", "text": "hello"}]})
    assert step.type == "assistant"
    assert step.content == "hello"


def test_parse_openai_tool_calls() -> None:
    step = parse_openai_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    assert step.type == "tool_calls"
    assert step.calls[0]["toolName"] == "read_file"
    assert step.calls[0]["input"] == {"path": "README.md"}


def test_to_openai_messages_includes_tool_result() -> None:
    converted = to_openai_messages(
        [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant_tool_call",
                "toolUseId": "1",
                "toolName": "read_file",
                "input": {"path": "a.txt"},
            },
            {
                "role": "tool_result",
                "toolUseId": "1",
                "content": "hi",
            },
        ]
    )
    assert converted[1]["tool_calls"][0]["id"] == "1"
    assert converted[2]["role"] == "tool"
    assert converted[2]["content"] == "hi"


class _FakeResponse:
    def __init__(self, raw: bytes) -> None:
        self._buf = BytesIO(raw)

    def readline(self) -> bytes:
        return self._buf.readline()

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)


def test_iter_sse_or_json_parses_events() -> None:
    raw = (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n'
        b"\n"
        b"data: [DONE]\n"
        b"\n"
    )
    events = list(iter_sse_or_json(_FakeResponse(raw)))
    assert events == [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}
    ]


def test_iter_sse_or_json_accepts_plain_json() -> None:
    raw = b'{"type":"message","content":[{"type":"text","text":"hello"}]}'
    events = list(iter_sse_or_json(_FakeResponse(raw)))
    assert events[0]["type"] == "message"
    assert events[0]["content"][0]["text"] == "hello"


def test_consume_anthropic_stream_text_and_tools() -> None:
    deltas: list[str] = []
    step = consume_anthropic_stream(
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Looking"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":"."}'},
            },
        ],
        deltas.append,
    )
    assert deltas == ["Looking"]
    assert step.type == "tool_calls"
    assert step.content == "Looking"
    assert step.calls[0]["toolName"] == "list_files"
    assert step.calls[0]["input"] == {"path": "."}


def test_consume_openai_stream_text() -> None:
    deltas: list[str] = []
    step = consume_openai_stream(
        [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        ],
        deltas.append,
    )
    assert deltas == ["Hel", "lo"]
    assert step.type == "assistant"
    assert step.content == "Hello"


def test_consume_openai_stream_tool_calls() -> None:
    step = consume_openai_stream(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "read_file", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"path":"a.txt"}'}}
                            ]
                        }
                    }
                ]
            },
        ]
    )
    assert step.type == "tool_calls"
    assert step.calls[0]["id"] == "call_1"
    assert step.calls[0]["toolName"] == "read_file"
    assert step.calls[0]["input"] == {"path": "a.txt"}
