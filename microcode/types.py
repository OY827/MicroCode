from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict, total=False):
    """One item in the conversation the model sees each turn."""

    role: Literal[
        "system",
        "user",
        "assistant",
        "assistant_tool_call",
        "tool_result",
    ]
    content: str
    toolUseId: str
    toolName: str
    input: Any
    isError: bool


class ToolCall(TypedDict):
    id: str
    toolName: str
    input: Any


@dataclass(slots=True)
class AgentStep:
    """What the model decided to do this round: talk, or call tools."""

    type: Literal["assistant", "tool_calls"]
    content: str = ""
    calls: list[ToolCall] = field(default_factory=list)


class ModelAdapter(Protocol):
    """Anything that can take messages and return the next step.

    A real LLM adapter and a mock adapter both implement this.
    The agent loop never cares which one it is.
    """

    def next(self, messages: list[ChatMessage]) -> AgentStep: ...
