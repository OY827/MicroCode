from __future__ import annotations

import sys
from typing import Any, Callable

from microcode.agent_loop import run_agent_turn
from microcode.checkpoint import WriteCheckpoint, undo_last_writes
from microcode.compact import CompactReport, compact_messages
from microcode.jobs import JobStore
from microcode.memory import ProjectMemory, apply_system_message, parse_memory_command
from microcode.permissions import PermissionStore, parse_permissions_command
from microcode.prompt import build_system_prompt
from microcode.session import Session, SessionStore, summarize_session, summarize_session_list
from microcode.skills import discover_skills, format_skill_list
from microcode.todos import TodoStore
from microcode.tooling import ToolRegistry
from microcode.types import ChatMessage, ModelAdapter

HELP_TEXT = """Local commands:
  /help      Show this help
  /session   Show the current saved session
  /sessions  List recent saved sessions
  /memory    Show project memory
  /memory add <note>  Append a project-memory note
  /skills    List local SKILL.md files
  /mcp       Show configured MCP servers
  /permissions           Show remembered allow rules
  /permissions clear     Forget all allow rules
  /permissions remove write <path>
  /permissions remove command <cmd>
  /todos     Show the in-memory task list
  /jobs      Show background shell jobs
  /compact   Shrink older tool results in this session
  /undo      Restore the last write group (files only)
  /exit      Leave the session
  /quit      Same as /exit
  Ctrl+C     Stop the current turn; at the prompt, cancel the line
"""

PERMISSIONS_USAGE = (
    "Usage: /permissions  |  /permissions clear  |  "
    "/permissions remove write <path>  |  /permissions remove command <cmd>"
)


def _handle_permissions_command(text: str, permissions: PermissionStore | None) -> str:
    if permissions is None:
        return "No permission store."
    action, kind, key = parse_permissions_command(text)
    if action == "show":
        return permissions.format()
    if action == "clear":
        permissions.clear()
        permissions.save()
        return "Cleared remembered permissions."
    if action == "remove":
        removed = False
        if kind == "write":
            removed = permissions.remove("write", key)
        elif kind == "command":
            removed = permissions.remove("command", key)
        if removed:
            permissions.save()
            return f"Removed {kind} {key}"
        return f"Not found: {kind} {key}"
    return PERMISSIONS_USAGE


def handle_local_command(text: str) -> str | None:
    """Return a local command name, or None if the line should go to the model."""

    command = text.strip().lower()
    if command in {"/exit", "/quit"}:
        return "exit"
    if command == "/help":
        return "help"
    if command == "/session":
        return "session"
    if command == "/sessions":
        return "sessions"
    if command == "/undo":
        return "undo"
    if command == "/compact":
        return "compact"
    if command == "/memory" or command.startswith("/memory "):
        return "memory"
    if command == "/skills":
        return "skills"
    if command == "/mcp":
        return "mcp"
    if command == "/permissions" or command.startswith("/permissions "):
        return "permissions"
    if command == "/todos":
        return "todos"
    if command == "/jobs":
        return "jobs"
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
    on_text_delta: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], Any] | None = None,
    on_ask_user: Callable[[str], str] | None = None,
    session: Session | None = None,
    store: SessionStore | None = None,
    compact_max_chars: int | None = None,
    on_compact: Callable[[CompactReport], None] | None = None,
    permissions: PermissionStore | None = None,
    todos: TodoStore | None = None,
    jobs: JobStore | None = None,
) -> list[ChatMessage]:
    """Append one user message and run the agent loop on the shared history."""

    next_messages = list(messages)
    next_messages.append({"role": "user", "content": user_text})
    current_checkpoint = WriteCheckpoint()
    next_messages = run_agent_turn(
        model=model,
        tools=tools,
        messages=next_messages,
        cwd=cwd,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_assistant_message=on_assistant_message,
        on_text_delta=on_text_delta,
        on_write_preview=on_write_preview,
        on_approve=on_approve,
        on_ask_user=on_ask_user,
        checkpoint=current_checkpoint,
        compact_max_chars=compact_max_chars,
        on_compact=on_compact,
        permissions=permissions,
        todos=todos,
        jobs=jobs,
    )
    if session is not None:
        session.messages = next_messages
        if current_checkpoint:
            session.checkpoint = current_checkpoint.to_list()
        if store is not None:
            store.save(session)
    return next_messages


def run_repl(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_write_preview: Callable[[str, str], None] | None = None,
    on_approve: Callable[[str], Any] | None = None,
    on_ask_user: Callable[[str], str] | None = None,
    read_line: Callable[[], str] | None = None,
    session: Session | None = None,
    store: SessionStore | None = None,
    compact_max_chars: int | None = None,
    on_compact: Callable[[CompactReport], None] | None = None,
    memory: ProjectMemory | None = None,
    system_prompt: str | None = None,
    mcp_status: str | None = None,
    permissions: PermissionStore | None = None,
    todos: TodoStore | None = None,
    jobs: JobStore | None = None,
) -> list[ChatMessage]:
    """Keep asking for user input until /exit. Reuses the same messages list."""

    read = read_line or (lambda: sys.stdin.readline())
    current = list(messages)
    print("Interactive session. Type /help or /exit.", file=sys.stderr)
    if session is not None:
        print(f"session: {session.id}", file=sys.stderr)

    try:
        while True:
            print("you> ", end="", file=sys.stderr, flush=True)
            try:
                raw = read()
            except KeyboardInterrupt:
                print(file=sys.stderr)
                continue
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
            if local == "session":
                if session is None:
                    print("No active session.", file=sys.stderr)
                else:
                    print(summarize_session(session), file=sys.stderr)
                continue
            if local == "sessions":
                if store is None:
                    print("No session store.", file=sys.stderr)
                else:
                    print(summarize_session_list(store.list_sessions()), file=sys.stderr)
                continue
            if local == "undo":
                if session is None:
                    print("No active session.", file=sys.stderr)
                else:
                    print(undo_last_writes(session=session, store=store, cwd=cwd), file=sys.stderr)
                continue
            if local == "compact":
                current, report = compact_messages(current, force=True)
                if session is not None:
                    session.messages = current
                    if store is not None:
                        store.save(session)
                print(report.summary(), file=sys.stderr)
                continue
            if local == "memory":
                if memory is None:
                    print("No project memory store.", file=sys.stderr)
                    continue
                action, note = parse_memory_command(text)
                if action == "add":
                    print(memory.append(note), file=sys.stderr)
                    if note.strip() and system_prompt is not None:
                        current = apply_system_message(current, build_system_prompt(system_prompt, cwd))
                        if session is not None:
                            session.messages = current
                            if store is not None:
                                store.save(session)
                elif action == "show":
                    print(memory.display(), file=sys.stderr)
                else:
                    print("Usage: /memory  or  /memory add <note>", file=sys.stderr)
                continue
            if local == "skills":
                print(format_skill_list(discover_skills(cwd)), file=sys.stderr)
                continue
            if local == "mcp":
                print(mcp_status or "No MCP servers. Add .microcode/mcp.json", file=sys.stderr)
                continue
            if local == "permissions":
                print(_handle_permissions_command(text, permissions), file=sys.stderr)
                continue
            if local == "todos":
                print(todos.format() if todos is not None else "No todo list.", file=sys.stderr)
                continue
            if local == "jobs":
                print(jobs.format() if jobs is not None else "No job list.", file=sys.stderr)
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
                on_text_delta=on_text_delta,
                on_write_preview=on_write_preview,
                on_approve=on_approve,
                on_ask_user=on_ask_user,
                session=session,
                store=store,
                compact_max_chars=compact_max_chars,
                on_compact=on_compact,
                permissions=permissions,
                todos=todos,
                jobs=jobs,
            )
        return current
    finally:
        if jobs is not None:
            report = jobs.kill_all()
            if report != "No running jobs.":
                print(report, file=sys.stderr)
