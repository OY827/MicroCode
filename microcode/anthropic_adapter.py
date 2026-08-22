from __future__ import annotations

from typing import Any

from microcode.api_client import post_json
from microcode.config import ModelConfig
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage, ToolCall


def anthropic_messages_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1/messages"):
        return url
    if url.endswith("/v1"):
        return f"{url}/messages"
    return f"{url}/v1/messages"


def to_anthropic_messages(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(str(message.get("content", "")))
            continue
        if role == "user":
            _push_block(converted, "user", {"type": "text", "text": str(message.get("content", ""))})
            continue
        if role == "assistant":
            _push_block(converted, "assistant", {"type": "text", "text": str(message.get("content", ""))})
            continue
        if role == "assistant_tool_call":
            _push_block(
                converted,
                "assistant",
                {
                    "type": "tool_use",
                    "id": message.get("toolUseId", ""),
                    "name": message.get("toolName", ""),
                    "input": message.get("input") or {},
                },
            )
            continue
        if role == "tool_result":
            _push_block(
                converted,
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("toolUseId", ""),
                    "content": str(message.get("content", "")),
                    "is_error": bool(message.get("isError", False)),
                },
            )

    return "\n\n".join(part for part in system_parts if part), converted


def _push_block(messages: list[dict[str, Any]], role: str, block: dict[str, Any]) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].append(block)
        return
    messages.append({"role": role, "content": [block]})


class AnthropicModelAdapter:
    """Anthropic Messages API, including DeepSeek's /anthropic endpoint."""

    def __init__(self, config: ModelConfig, tools: ToolRegistry) -> None:
        self.config = config
        self.tools = tools

    def next(self, messages: list[ChatMessage]) -> AgentStep:
        system, converted = to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": converted,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in self.tools.list()
            ],
            "thinking": {"type": "disabled"},
        }
        if system:
            payload["system"] = system

        data = post_json(
            anthropic_messages_url(self.config.base_url),
            payload,
            {
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.config.api_key,
            },
        )
        return parse_anthropic_response(data)


def parse_anthropic_response(data: dict[str, Any]) -> AgentStep:
    text_parts: list[str] = []
    calls: list[ToolCall] = []

    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "tool_use":
            calls.append(
                {
                    "id": str(block.get("id") or "tool-1"),
                    "toolName": str(block.get("name") or ""),
                    "input": block.get("input") or {},
                }
            )

    content = "\n".join(text_parts).strip()
    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls)
    return AgentStep(type="assistant", content=content)
