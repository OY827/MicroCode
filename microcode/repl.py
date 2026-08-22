from __future__ import annotations

import sys
from typing import Callable

from microcode.agent_loop import run_agent_turn
from microcode.tooling import ToolRegistry
from microcode.types import ChatMessage, ModelAdapter

HELP_TEXT = """Local commands:
  /help   Show this help
  /exit   Leave the session
  /quit   Same as /exit
"""


def handle_local_command(text: str) -> str | None:
    """Return 'exit', 'help', or None if the line should go to the model."""

    command = text.strip().lower()
    if command in {"/exit", "/quit"}:
        return "exit"
    if command == "/help":
        return "help"
    return None


def run_user_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    user_text: str,
    cwd: str,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], bool] | None = None,
) -> list[ChatMessage]:
    """Append one user message and run the agent loop on the shared history."""

    next_messages = list(messages)
    next_messages.append({"role": "user", "content": user_text})
    return run_agent_turn(
        model=model,
        tools=tools,
        messages=next_messages,
        cwd=cwd,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_assistant_message=on_assistant_message,
        on_write_preview=on_write_preview,
        on_approve=on_approve,
    )


def run_repl(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], bool] | None = None,
    read_line: Callable[[], str] | None = None,
) -> list[ChatMessage]:
    """Keep asking for user input until /exit. Reuses the same messages list."""

    read = read_line or (lambda: sys.stdin.readline())
    current = list(messages)
    print("Interactive session. Type /help or /exit.", file=sys.stderr)

    while True:
        print("you> ", end="", file=sys.stderr, flush=True)
        try:
            raw = read()
        except KeyboardInterrupt:
            print(file=sys.stderr)
            break
        if raw == "":
            print(file=sys.stderr)
            break

        text = raw.strip()
        if not text:
            continue

        local = handle_local_command(text)
        if local == "exit":
            print("bye", file=sys.stderr)
            break
        if local == "help":
            print(HELP_TEXT, end="", file=sys.stderr)
            continue

        current = run_user_turn(
            model=model,
            tools=tools,
            messages=current,
            user_text=text,
            cwd=cwd,
            on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
            on_assistant_message=on_assistant_message,
            on_write_preview=on_write_preview,
            on_approve=on_approve,
        )
    return current
