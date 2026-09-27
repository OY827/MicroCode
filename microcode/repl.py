from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

from microcode.agent_loop import run_agent_turn
from microcode.checkpoint import (
    WriteCheckpoint,
    format_checkpoints,
    parse_rewind_target,
    preview_rewind,
    redo_writes,
    remember_write_group,
    rewind_writes,
    undo_last_writes,
)
from microcode.compact import CompactReport, compact_messages
from microcode.fixture_export import export_fixture
from microcode.jobs import JobStore
from microcode.memory import ProjectMemory, apply_system_message, parse_memory_command
from microcode.permissions import (
    PermissionStore,
    format_permission_mode,
    parse_mode_command,
    parse_permissions_command,
)
from microcode.models import apply_model_switch, describe_model, model_label
from microcode.project_init import run_init
from microcode.prompt import build_system_prompt
from microcode.readiness import build_readiness_report
from microcode.session import Session, SessionStore, format_replay, summarize_session, summarize_session_list
from microcode.session_note import finish_session_memory
from microcode.status import format_status
from microcode.shortcuts import execute_tool_shortcut, parse_tool_shortcut
from microcode.skills import discover_skills, format_skill_list
from microcode.todos import TodoStore
from microcode.tooling import ToolRegistry
from microcode.turn_tape import format_tape, open_turn_tape, parse_tape_args, tape_path
from microcode.types import ChatMessage, ModelAdapter
from microcode.usage import UsageLedger

HELP_TEXT = """Local commands:
  /help      Show this help
  /status    Snapshot of this chat: model, size, memory, undo
  /replay [id]  Replay this chat, or a saved session id
  /tape [n|path]  Last n lines of the turn tape, or where the jsonl file is
  /fixture   Save this chat as an offline pytest replay
  /model     Show the current model
  /model <name>|mock|live  Switch model for the next turn
  /cost      Tokens and estimated spend for this chat
  /session   Show the current saved session
  /sessions  List recent saved sessions
  /memory    Show project memory
  /memory add <note>  Append a project-memory note
  /memory add user:|local: <note>  Write to user or local scope
  /memory search <query>  Rank notes with BM25 + TF-IDF (+ MiniLM if installed)
  /memory maintain        Apply working/short-term/long-term/archival rules
  /skills    List local SKILL.md files
  /mcp       Show configured MCP servers
  /permissions           Show remembered allow rules
  /permissions clear     Forget all allow rules
  /permissions remove write <path>
  /permissions remove command <cmd>
  /todos     Show the in-memory task list
  /jobs      Show background shell jobs
  /readiness Local checkup: keys, address, backup key (no network)
  /init      Scan the repo: how to test, what not to touch
  /compact   Shrink older tool results; fat output goes to .microcode/tool-results/
  /undo      Restore the last write group (files only)
  /redo      Re-apply the last undo or rewind
  /checkpoints              List rewind checkpoints
  /rewind-preview [id]      Show which files a rewind would restore
  /rewind [id]              Restore files from that checkpoint onward
  /mode      Show this session's permission mode
  /mode ask|yes|read        Prompt, auto-approve, or block writes and commands
  /ls [path]                List one folder (no model)
  /tree [path][::depth]     Show a directory tree (no model)
  /grep <pattern>[::path]   Search files (no model)
  /read <path>              Read a file (no model)
  /write <path>::<text>     Create or overwrite a file (no model)
  /edit <path>::<old>::<new>  Replace one unique string (no model)
  /patch <path>::<old>=><new>[::...]  Several replacements (no model)
  /copy <src>::<dst>        Copy a file or folder (no model)
  /move <src>::<dst>        Rename or move (no model)
  /delete <path>[::recursive]  Delete a file or folder (no model)
  /cmd <command>            Run a shell command (no model)
  /git status|diff|log      Inspect git (no model)
  /git add <path> [path...] Stage files (no model)
  /git commit <message>     Commit staged files (no model)
  /exit      Leave the session
  /quit      Same as /exit
  Ctrl+C     Stop the current turn; at the prompt, cancel the line

TUI keys (full-screen session):
  Enter send   Esc clear line   ↑/↓ history   PgUp/PgDn scroll
  Tab complete slash command   Ctrl+C leave
"""

def _handle_replay(
    text: str,
    *,
    session: Session | None,
    store: SessionStore | None,
    messages: list[ChatMessage],
) -> str:
    arg = text[len("/replay") :].strip()
    if not arg:
        return format_replay(messages=messages, session=session)
    if store is None:
        return "No session store.\n"
    target = None if arg.lower() == "latest" else arg
    try:
        loaded = store.load(target)
    except FileNotFoundError as error:
        return f"{error}\n"
    return format_replay(messages=loaded.messages, session=loaded)


PERMISSIONS_USAGE = (
    "Usage: /permissions  |  /permissions clear  |  "
    "/permissions remove write <path>  |  /permissions remove command <cmd>"
)
MODE_USAGE = "Usage: /mode  |  /mode ask  |  /mode yes  |  /mode read"


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


def _handle_mode_command(text: str, session: Session | None, store: SessionStore | None) -> str:
    if session is None:
        return "No active session."
    action, mode = parse_mode_command(text)
    if action == "show":
        return format_permission_mode(session.permission_mode)
    if action == "set":
        session.permission_mode = mode
        if store is not None:
            store.save(session)
        return format_permission_mode(session.permission_mode)
    return MODE_USAGE


def handle_local_command(text: str) -> str | None:
    """Return a local command name, or None if the line should go to the model."""

    command = text.strip().lower()
    if command in {"/exit", "/quit"}:
        return "exit"
    if command == "/help":
        return "help"
    if command == "/status":
        return "status"
    if command == "/replay" or command.startswith("/replay "):
        return "replay"
    if command == "/tape" or command.startswith("/tape "):
        return "tape"
    if command == "/fixture":
        return "fixture"
    if command == "/model" or command.startswith("/model "):
        return "model"
    if command == "/cost":
        return "cost"
    if command == "/session":
        return "session"
    if command == "/sessions":
        return "sessions"
    if command == "/undo":
        return "undo"
    if command == "/redo":
        return "redo"
    if command == "/mode" or command.startswith("/mode "):
        return "mode"
    if command == "/checkpoints":
        return "checkpoints"
    if command == "/rewind-preview" or command.startswith("/rewind-preview "):
        return "rewind-preview"
    if command == "/rewind" or command.startswith("/rewind "):
        return "rewind"
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
    if command == "/readiness":
        return "readiness"
    if command == "/init":
        return "init"
    return None


@dataclass
class LocalCommandResult:
    handled: bool
    exit: bool = False
    output: str = ""
    messages: list[ChatMessage] | None = None
    model: ModelAdapter | None = None


def apply_local_command(
    text: str,
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    session: Session | None = None,
    store: SessionStore | None = None,
    memory: ProjectMemory | None = None,
    system_prompt: str | None = None,
    mcp_status: str | None = None,
    permissions: PermissionStore | None = None,
    todos: TodoStore | None = None,
    jobs: JobStore | None = None,
    usage: UsageLedger | None = None,
) -> LocalCommandResult:
    """Run a local slash command. Does not talk to the model."""

    local = handle_local_command(text)
    if local is None:
        return LocalCommandResult(handled=False)
    if local == "exit":
        return LocalCommandResult(handled=True, exit=True, output="bye")
    current = list(messages)
    current_model = model
    ledger = usage
    if ledger is None:
        ledger = UsageLedger.from_dict(session.usage if session is not None else None)

    if local == "help":
        return LocalCommandResult(handled=True, output=HELP_TEXT)
    if local == "status":
        return LocalCommandResult(
            handled=True,
            output=format_status(
                cwd=cwd,
                messages=current,
                session=session,
                memory=memory,
                permissions=permissions,
                model=current_model,
                usage=ledger,
            ),
        )
    if local == "replay":
        return LocalCommandResult(
            handled=True,
            output=_handle_replay(text, session=session, store=store, messages=current),
        )
    if local == "tape":
        action, limit = parse_tape_args(text)
        if action == "path":
            return LocalCommandResult(handled=True, output=str(tape_path(cwd)))
        if action == "show":
            return LocalCommandResult(handled=True, output=format_tape(cwd, limit=limit))
        return LocalCommandResult(handled=True, output="Usage: /tape  |  /tape <n>  |  /tape path")
    if local == "fixture":
        return LocalCommandResult(
            handled=True,
            output=export_fixture(
                cwd,
                current,
                session_id=session.id if session is not None else "",
            ),
        )
    if local == "model":
        arg = text[len("/model") :].strip()
        if not arg:
            return LocalCommandResult(handled=True, output=describe_model(current_model))
        current_model, message = apply_model_switch(current_model, tools, arg)
        if session is not None:
            session.model_name = model_label(current_model)
            if store is not None:
                store.save(session)
        return LocalCommandResult(handled=True, output=message, model=current_model)
    if local == "cost":
        return LocalCommandResult(
            handled=True,
            output=ledger.format(model_name=model_label(current_model), messages=current),
        )
    if local == "session":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(handled=True, output=summarize_session(session))
    if local == "sessions":
        if store is None:
            return LocalCommandResult(handled=True, output="No session store.")
        return LocalCommandResult(handled=True, output=summarize_session_list(store.list_sessions()))
    if local == "undo":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(
            handled=True, output=undo_last_writes(session=session, store=store, cwd=cwd)
        )
    if local == "redo":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(handled=True, output=redo_writes(session=session, store=store, cwd=cwd))
    if local == "mode":
        return LocalCommandResult(handled=True, output=_handle_mode_command(text, session, store))
    if local == "checkpoints":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(handled=True, output=format_checkpoints(session))
    if local == "rewind-preview":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(
            handled=True,
            output=preview_rewind(session=session, target_id=parse_rewind_target(text)),
        )
    if local == "rewind":
        if session is None:
            return LocalCommandResult(handled=True, output="No active session.")
        return LocalCommandResult(
            handled=True,
            output=rewind_writes(
                session=session,
                store=store,
                cwd=cwd,
                target_id=parse_rewind_target(text),
            ),
        )
    if local == "compact":
        current, report = compact_messages(current, force=True, cwd=cwd)
        if session is not None:
            session.messages = current
            if store is not None:
                store.save(session)
        return LocalCommandResult(handled=True, output=report.summary(), messages=current)
    if local == "memory":
        if memory is None:
            return LocalCommandResult(handled=True, output="No project memory store.")
        action, note = parse_memory_command(text)
        if action == "add":
            output = memory.append(note)
            if note.strip() and system_prompt is not None:
                current = apply_system_message(
                    current, build_system_prompt(system_prompt, cwd, memory=memory)
                )
                if session is not None:
                    session.messages = current
                    if store is not None:
                        store.save(session)
                return LocalCommandResult(handled=True, output=output, messages=current)
            return LocalCommandResult(handled=True, output=output)
        if action == "show":
            return LocalCommandResult(handled=True, output=memory.display())
        if action == "search":
            return LocalCommandResult(handled=True, output=memory.format_search(note))
        if action == "maintain":
            return LocalCommandResult(handled=True, output=memory.maintain())
        return LocalCommandResult(
            handled=True,
            output=(
                "Usage: /memory  |  /memory add [user:|project:|local:] <note>  |  "
                "/memory search <query>  |  /memory maintain"
            ),
        )
    if local == "skills":
        return LocalCommandResult(handled=True, output=format_skill_list(discover_skills(cwd)))
    if local == "mcp":
        return LocalCommandResult(
            handled=True,
            output=mcp_status or "No MCP servers. Add .microcode/mcp.json",
        )
    if local == "permissions":
        return LocalCommandResult(handled=True, output=_handle_permissions_command(text, permissions))
    if local == "todos":
        return LocalCommandResult(
            handled=True,
            output=todos.format() if todos is not None else "No todo list.",
        )
    if local == "jobs":
        return LocalCommandResult(
            handled=True,
            output=jobs.format() if jobs is not None else "No job list.",
        )
    if local == "readiness":
        return LocalCommandResult(handled=True, output=build_readiness_report().format())
    if local == "init":
        report = run_init(cwd, memory)
        if memory is not None and system_prompt is not None:
            current = apply_system_message(
                current, build_system_prompt(system_prompt, cwd, memory=memory)
            )
            if session is not None:
                session.messages = current
                if store is not None:
                    store.save(session)
            return LocalCommandResult(handled=True, output=report, messages=current)
        return LocalCommandResult(handled=True, output=report)
    return LocalCommandResult(handled=True, output="Unknown command. Type /help.")


def _print_local(text: str) -> None:
    if not text:
        return
    if text.endswith("\n"):
        print(text, end="", file=sys.stderr)
    else:
        print(text, file=sys.stderr)


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
    on_revise_write: Callable[[str, str, str], Any] | None = None,
    on_ask_user: Callable[[str], str] | None = None,
    session: Session | None = None,
    store: SessionStore | None = None,
    compact_max_chars: int | None = None,
    on_compact: Callable[[CompactReport], None] | None = None,
    permissions: PermissionStore | None = None,
    todos: TodoStore | None = None,
    jobs: JobStore | None = None,
    system_prompt: str | None = None,
    memory: ProjectMemory | None = None,
    usage: UsageLedger | None = None,
) -> list[ChatMessage]:
    """Append one user message and run the agent loop on the shared history."""

    next_messages = list(messages)
    if system_prompt is not None:
        next_messages = apply_system_message(
            next_messages,
            build_system_prompt(system_prompt, cwd, query=user_text, memory=memory),
        )
    next_messages.append({"role": "user", "content": user_text})
    tape = open_turn_tape(cwd, session.id if session is not None else "")
    if tape is not None:
        tape.record_user(user_text)
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
        on_revise_write=on_revise_write,
        on_ask_user=on_ask_user,
        checkpoint=current_checkpoint,
        compact_max_chars=compact_max_chars,
        on_compact=on_compact,
        permissions=permissions,
        todos=todos,
        jobs=jobs,
        usage=usage,
        permission_mode=session.permission_mode if session is not None else "ask",
        tape=tape,
    )
    if session is not None:
        session.messages = next_messages
        session.model_name = model_label(model)
        if usage is not None:
            session.usage = usage.to_dict()
        if current_checkpoint:
            remember_write_group(session, current_checkpoint)
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
    on_revise_write: Callable[[str, str, str], Any] | None = None,
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
    usage: UsageLedger | None = None,
) -> list[ChatMessage]:
    """Keep asking for user input until /exit. Reuses the same messages list."""

    read = read_line or (lambda: sys.stdin.readline())
    current = list(messages)
    current_model = model
    ledger = usage
    if ledger is None:
        ledger = UsageLedger.from_dict(session.usage if session is not None else None)
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
                if memory is not None:
                    print(finish_session_memory(memory, messages=current, session=session), file=sys.stderr)
                print(file=sys.stderr)
                break

            text = raw.strip()
            if not text:
                continue

            local = apply_local_command(
                text,
                model=current_model,
                tools=tools,
                messages=current,
                cwd=cwd,
                session=session,
                store=store,
                memory=memory,
                system_prompt=system_prompt,
                mcp_status=mcp_status,
                permissions=permissions,
                todos=todos,
                jobs=jobs,
                usage=ledger,
            )
            if local.handled:
                if local.messages is not None:
                    current = local.messages
                if local.model is not None:
                    current_model = local.model
                if local.exit:
                    if memory is not None:
                        print(
                            finish_session_memory(memory, messages=current, session=session),
                            file=sys.stderr,
                        )
                    _print_local(local.output)
                    break
                _print_local(local.output)
                continue

            shortcut = parse_tool_shortcut(text)
            if isinstance(shortcut, str):
                print(shortcut, file=sys.stderr)
                continue
            if shortcut is not None:
                result = execute_tool_shortcut(
                    shortcut,
                    tools=tools,
                    cwd=cwd,
                    on_write_preview=on_write_preview,
                    on_approve=on_approve,
                    on_revise_write=on_revise_write,
                    session=session,
                    store=store,
                    permissions=permissions,
                    jobs=jobs,
                )
                flag = "ok" if result.ok else "error"
                print(f"{shortcut.tool_name} [{flag}]\n{result.output}", file=sys.stderr)
                continue

            current = run_user_turn(
                model=current_model,
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
                on_revise_write=on_revise_write,
                on_ask_user=on_ask_user,
                session=session,
                store=store,
                compact_max_chars=compact_max_chars,
                on_compact=on_compact,
                permissions=permissions,
                todos=todos,
                jobs=jobs,
                system_prompt=system_prompt,
                memory=memory,
                usage=ledger,
            )
        return current
    finally:
        if jobs is not None:
            report = jobs.kill_all()
            if report != "No running jobs.":
                print(report, file=sys.stderr)
