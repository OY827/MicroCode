from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from microcode.compact import CompactReport, compact_limit, compact_messages
from microcode.tooling import ToolContext, ToolRegistry
from microcode.types import ChatMessage, ModelAdapter, ToolCall

INTERRUPTED_MESSAGE = "Interrupted by user."

if TYPE_CHECKING:
    from microcode.checkpoint import WriteCheckpoint
    from microcode.jobs import JobStore
    from microcode.permissions import PermissionStore
    from microcode.todos import TodoStore


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
    on_text_delta: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], Any] | None = None,
    on_ask_user: Callable[[str], str] | None = None,
    checkpoint: WriteCheckpoint | None = None,
    compact_max_chars: int | None = None,
    on_compact: Callable[[CompactReport], None] | None = None,
    permissions: PermissionStore | None = None,
    todos: TodoStore | None = None,
    jobs: JobStore | None = None,
) -> list[ChatMessage]:
    """One user task: model <-> tools until the model answers or hits max_steps.

    This is the whole product in step 1. UI, memory, and real APIs plug in later.
    """

    current_messages = list(messages)
    max_chars = compact_limit() if compact_max_chars is None else compact_max_chars

    for _ in range(max_steps):
        current_messages, report = compact_messages(current_messages, max_chars=max_chars)
        if report.compacted and on_compact:
            on_compact(report)

        streamed = False

        def emit_delta(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            if on_text_delta:
                on_text_delta(chunk)

        try:
            next_step = model.next(current_messages, on_text_delta=emit_delta if on_text_delta else None)
        except KeyboardInterrupt:
            return _interrupt_turn(
                current_messages,
                on_assistant_message=on_assistant_message,
                on_text_delta=on_text_delta,
                streamed=streamed,
            )
        except Exception as error:  # noqa: BLE001
            fallback = f"Model API error ({type(error).__name__}): {error}"
            if on_assistant_message:
                on_assistant_message(fallback)
            current_messages.append({"role": "assistant", "content": fallback})
            return current_messages

        if streamed and on_text_delta:
            on_text_delta("\n")

        if next_step.type != "tool_calls" or not next_step.calls:
            if not streamed and on_assistant_message:
                on_assistant_message(next_step.content)
            current_messages.append({"role": "assistant", "content": next_step.content})
            return current_messages

        for call in next_step.calls:
            try:
                if on_tool_start:
                    on_tool_start(call["toolName"], call["input"])
                result = tools.execute(
                    call["toolName"],
                    call["input"],
                    ToolContext(
                        cwd=cwd,
                        on_write_preview=on_write_preview,
                        on_approve=on_approve,
                        on_ask_user=on_ask_user,
                        checkpoint=checkpoint,
                        permissions=permissions,
                        todos=todos,
                        jobs=jobs,
                        model=model,
                        on_tool_start=on_tool_start,
                        on_tool_result=on_tool_result,
                    ),
                )
            except KeyboardInterrupt:
                return _interrupt_turn(
                    current_messages,
                    on_assistant_message=on_assistant_message,
                    on_text_delta=on_text_delta,
                    streamed=False,
                    pending=call,
                    on_tool_result=on_tool_result,
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


def _interrupt_turn(
    messages: list[ChatMessage],
    *,
    on_assistant_message: Callable[[str], None] | None,
    on_text_delta: Callable[[str], None] | None,
    streamed: bool,
    pending: ToolCall | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
) -> list[ChatMessage]:
    """Close the current turn after Ctrl+C so the REPL can keep the history."""

    if pending is not None:
        messages.append(
            {
                "role": "assistant_tool_call",
                "toolUseId": pending["id"],
                "toolName": pending["toolName"],
                "input": pending["input"],
            }
        )
        messages.append(
            {
                "role": "tool_result",
                "toolUseId": pending["id"],
                "toolName": pending["toolName"],
                "content": INTERRUPTED_MESSAGE,
                "isError": True,
            }
        )
        if on_tool_result:
            on_tool_result(pending["toolName"], INTERRUPTED_MESSAGE, True)
    if streamed and on_text_delta:
        on_text_delta("\n")
    if on_assistant_message:
        on_assistant_message(INTERRUPTED_MESSAGE)
    messages.append({"role": "assistant", "content": INTERRUPTED_MESSAGE})
    return messages
