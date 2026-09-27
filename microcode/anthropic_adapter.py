from __future__ import annotations

import json
from typing import Any, Callable

from microcode.api_client import post_json, post_sse
from microcode.config import ModelConfig
from microcode.tooling import ToolRegistry
from microcode.types import AgentStep, ChatMessage, TokenUsage, ToolCall
from microcode.usage import read_anthropic_stream_usage, usage_from_anthropic


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

    def __init__(self, config: ModelConfig, tools: ToolRegistry, *, stream: bool = True) -> None:
        self.config = config
        self.tools = tools
        self.stream = stream

    def next(
        self,
        messages: list[ChatMessage],
        on_text_delta: Callable[[str], None] | None = None,
    ) -> AgentStep:
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

        url = anthropic_messages_url(self.config.base_url)
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.config.api_key,
        }
        if self.stream:
            headers["accept"] = "text/event-stream"
        if self.stream:
            payload["stream"] = True
            return consume_anthropic_stream(post_sse(url, payload, headers), on_text_delta)
        return parse_anthropic_response(post_json(url, payload, headers))


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
    usage = usage_from_anthropic(data)
    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls, usage=usage)
    return AgentStep(type="assistant", content=content, usage=usage)


def consume_anthropic_stream(
    events,
    on_text_delta: Callable[[str], None] | None = None,
) -> AgentStep:
    """Turn Anthropic SSE events (or one complete message) into an AgentStep."""

    texts: dict[int, str] = {}
    tools: dict[int, dict[str, Any]] = {}
    input_tokens: int | None = None
    output_tokens: int | None = None

    for event in events:
        if not isinstance(event, dict):
            continue
        inp, out = read_anthropic_stream_usage(event)
        if inp is not None:
            input_tokens = inp
        if out is not None:
            output_tokens = out
        if event.get("type") == "message" and isinstance(event.get("content"), list):
            step = parse_anthropic_response(event)
            if on_text_delta and step.content:
                on_text_delta(step.content)
            return step
        _apply_anthropic_event(event, texts, tools, on_text_delta)

    text_parts = [texts[index] for index in sorted(texts) if texts[index]]
    calls: list[ToolCall] = []
    for index in sorted(tools):
        item = tools[index]
        parsed_input = item.get("input") or {}
        raw_json = item.get("json") or ""
        if raw_json:
            try:
                loaded = json.loads(raw_json)
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                parsed_input = loaded
        calls.append(
            {
                "id": str(item.get("id") or "tool-1"),
                "toolName": str(item.get("name") or ""),
                "input": parsed_input if isinstance(parsed_input, dict) else {},
            }
        )

    content = "\n".join(text_parts).strip()
    usage = None
    if input_tokens or output_tokens:
        usage = TokenUsage(input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)
    if calls:
        return AgentStep(type="tool_calls", content=content, calls=calls, usage=usage)
    return AgentStep(type="assistant", content=content, usage=usage)


def _apply_anthropic_event(
    event: dict[str, Any],
    texts: dict[int, str],
    tools: dict[int, dict[str, Any]],
    on_text_delta: Callable[[str], None] | None,
) -> None:
    event_type = event.get("type")
    index = int(event.get("index") or 0)
    if event_type == "content_block_start":
        block = event.get("content_block") or {}
        if block.get("type") == "text":
            texts[index] = str(block.get("text") or "")
        elif block.get("type") == "tool_use":
            tools[index] = {
                "id": block.get("id") or "tool-1",
                "name": block.get("name") or "",
                "json": "",
                "input": block.get("input") if isinstance(block.get("input"), dict) else {},
            }
        return
    if event_type != "content_block_delta":
        return
    delta = event.get("delta") or {}
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = str(delta.get("text") or "")
        texts[index] = texts.get(index, "") + text
        if text and on_text_delta:
            on_text_delta(text)
        return
    if delta_type == "input_json_delta" and index in tools:
        tools[index]["json"] += str(delta.get("partial_json") or "")
