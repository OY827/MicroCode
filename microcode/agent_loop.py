from __future__ import annotations

from typing import Callable

from microcode.tooling import ToolContext, ToolRegistry
from microcode.types import ChatMessage, ModelAdapter


def run_agent_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    max_steps: int = 20,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], bool] | None = None,
) -> list[ChatMessage]:
    """One user task: model <-> tools until the model answers or hits max_steps.

    This is the whole product in step 1. UI, memory, and real APIs plug in later.
    """

    current_messages = list(messages)

    for _ in range(max_steps):
        try:
            next_step = model.next(current_messages)
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001
            fallback = f"Model API error ({type(error).__name__}): {error}"
            if on_assistant_message:
                on_assistant_message(fallback)
            current_messages.append({"role": "assistant", "content": fallback})
            return current_messages

        if next_step.type != "tool_calls" or not next_step.calls:
            if on_assistant_message:
                on_assistant_message(next_step.content)
            current_messages.append({"role": "assistant", "content": next_step.content})
            return current_messages

        for call in next_step.calls:
            if on_tool_start:
                on_tool_start(call["toolName"], call["input"])

            result = tools.execute(
                call["toolName"],
                call["input"],
                ToolContext(cwd=cwd, on_write_preview=on_write_preview, on_approve=on_approve),
            )

            if on_tool_result:
                on_tool_result(call["toolName"], result.output, not result.ok)

            current_messages.append(
                {
                    "role": "assistant_tool_call",
                    "toolUseId": call["id"],
                    "toolName": call["toolName"],
                    "input": call["input"],
                }
            )
            current_messages.append(
                {
                    "role": "tool_result",
                    "toolUseId": call["id"],
                    "toolName": call["toolName"],
                    "content": result.output,
                    "isError": not result.ok,
                }
            )

    fallback = "Reached the maximum tool step limit for this turn."
    if on_assistant_message:
        on_assistant_message(fallback)
    current_messages.append({"role": "assistant", "content": fallback})
    return current_messages
