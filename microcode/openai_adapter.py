from __future__ import annotations

import json
from typing import Any

from microcode.api_client import post_json
from microcode.config import ModelConfig
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage, ToolCall


def openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            converted.append({"role": "system", "content": str(message.get("content", ""))})
            continue
        if role == "user":
            converted.append({"role": "user", "content": str(message.get("content", ""))})
            continue
        if role == "assistant":
            converted.append({"role": "assistant", "content": str(message.get("content", ""))})
            continue
        if role == "assistant_tool_call":
            converted.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": message.get("toolUseId", ""),
                            "type": "function",
                            "function": {
                                "name": message.get("toolName", ""),
                                "arguments": json.dumps(message.get("input") or {}),
                            },
                        }
                    ],
                }
            )
            continue
        if role == "tool_result":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("toolUseId", ""),
                    "content": str(message.get("content", "")),
                }
            )
    return converted


class OpenAIModelAdapter:
    """OpenAI Chat Completions API, including DeepSeek's default endpoint."""

    def __init__(self, config: ModelConfig, tools: ToolRegistry) -> None:
        self.config = config
        self.tools = tools

    def next(self, messages: list[ChatMessage]) -> AgentStep:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": to_openai_messages(messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in self.tools.list()
            ],
        }
        data = post_json(
            openai_chat_url(self.config.base_url),
            payload,
            {
                "content-type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        return parse_openai_response(data)


def parse_openai_response(data: dict[str, Any]) -> AgentStep:
    choices = data.get("choices") or []
    if not choices:
        return AgentStep(type="assistant", content="")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        try:
            parsed_input = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            parsed_input = {}
        if not isinstance(parsed_input, dict):
            parsed_input = {}
        calls.append(
            {
                "id": str(raw.get("id") or "tool-1"),
                "toolName": str(function.get("name") or ""),
                "input": parsed_input,
            }
        )

    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls)
    return AgentStep(type="assistant", content=content)
