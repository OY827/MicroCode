from __future__ import annotations

import json
from typing import Any, Callable

from microcode.api_client import post_json, post_sse
from microcode.config import ModelConfig
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage, ToolCall
from microcode.usage import usage_from_openai


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

    def __init__(self, config: ModelConfig, tools: ToolRegistry, *, stream: bool = True) -> None:
        self.config = config
        self.tools = tools
        self.stream = stream

    def next(
        self,
        messages: list[ChatMessage],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> AgentStep:
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
        url = openai_chat_url(self.config.base_url)
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if self.stream:
            headers["accept"] = "text/event-stream"
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            return consume_openai_stream(post_sse(url, payload, headers), on_text_delta)
        return parse_openai_response(post_json(url, payload, headers))


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

    usage = usage_from_openai(data)
    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls, usage=usage)
    return AgentStep(type="assistant", content=content, usage=usage)


def consume_openai_stream(
    events,
    on_text_delta: Callable[[str], None] | None = None,
) -> AgentStep:
    """Turn OpenAI SSE chunks (or one complete completion) into an AgentStep."""

    content_parts: list[str] = []
    tools: dict[int, dict[str, str]] = {}
    usage = None

    for event in events:
        if not isinstance(event, dict):
            continue
        found = usage_from_openai(event)
        if found is not None:
            usage = found
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if isinstance(choice.get("message"), dict) and "delta" not in choice:
            return parse_openai_response(event)
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            content_parts.append(text)
            if on_text_delta:
                on_text_delta(text)
        for raw in delta.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            index = int(raw.get("index") or 0)
            entry = tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if raw.get("id"):
                entry["id"] = str(raw["id"])
            function = raw.get("function") or {}
            if function.get("name"):
                entry["name"] += str(function["name"])
            if function.get("arguments"):
                entry["arguments"] += str(function["arguments"])

    content = "".join(content_parts).strip()
    calls: list[ToolCall] = []
    for index in sorted(tools):
        entry = tools[index]
        try:
            parsed_input = json.loads(entry.get("arguments") or "{}")
        except json.JSONDecodeError:
            parsed_input = {}
        if not isinstance(parsed_input, dict):
            parsed_input = {}
        calls.append(
            {
                "id": entry.get("id") or "tool-1",
                "toolName": entry.get("name") or "",
                "input": parsed_input,
            }
        )
    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls, usage=usage)
    return AgentStep(type="assistant", content=content, usage=usage)
