from microcode.anthropic_adapter import (
    anthropic_messages_url,
    parse_anthropic_response,
    to_anthropic_messages,
)
from microcode.config import ModelConfig, resolve_protocol
from microcode.models import create_model_adapter
from microcode.openai_adapter import openai_chat_url, parse_openai_response, to_openai_messages
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
