from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcode.config import load_dotenv, load_model_config, resolve_protocol
from microcode.mcp import connect_mcp
from microcode.memory import ProjectMemory, apply_system_message
from microcode.mock_model import MockModelAdapter
from microcode.models import create_model_adapter
from microcode.permissions import PermissionStore
from microcode.prompt import build_system_prompt
from microcode.repl import run_repl, run_user_turn
from microcode.session import SessionStore
from microcode.skills import discover_skills
from microcode.todos import TodoStore
from microcode.jobs import JobStore
from microcode.tools import create_default_tool_registry

SYSTEM_PROMPT = (
    "You are MicroCode, a local coding agent working in a real repository. "
    "Use list_files, grep_files, and read_file to inspect the workspace before changing it. "
    "Use edit_file for unique search/replace edits and write_file to create or overwrite files. "
    "Use run_command to execute tests or scripts in the workspace after making changes. "
    "For long-running commands, set background=true and later collect output with await_job. "
    "If local skills are listed below, use load_skill to load the matching SKILL.md before following it. "
    "For tasks with more than two steps, use todo_write to keep a short checklist and update it as you go. "
    "If a key choice is unspecified and guessing would change the code, use ask_user before editing. "
    "Use web_fetch to read public documentation URLs; do not use it for local or private addresses. "
    "For broad codebase questions, use explore so a read-only sub-agent searches and returns a summary. "
    "Prefer reading files over guessing. When the task is done, give a concise final answer."
)


def _print_tool_start(name: str, args: dict) -> None:
    print(f"→ tool {name} {args}", file=sys.stderr)


def _print_tool_result(name: str, output: str, is_error: bool) -> None:
    flag = "error" if is_error else "ok"
    preview = output if len(output) <= 400 else output[:400] + "\n..."
    print(f"← {name} [{flag}]\n{preview}\n", file=sys.stderr)


def _print_write_preview(path: str, diff: str) -> None:
    print(f"✎ write preview {path}\n{diff}\n", file=sys.stderr)


def _print_compact(report) -> None:
    print(report.summary(), file=sys.stderr)


def _ask_approval(summary: str) -> bool | str:
    print(f"? {summary}", file=sys.stderr)
    print("Approve? [y/N/a] ", end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if not line:
        return False
    text = line.strip().lower()
    if text in {"a", "always"}:
        return "always"
    return text in {"y", "yes"}


def _ask_user(question: str) -> str:
    print(f"? {question}", file=sys.stderr)
    print("answer> ", end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if not line:
        return ""
    return line.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MicroCode coding agent")
    parser.add_argument("prompt", nargs="?", help="Optional first user task")
    parser.add_argument(
        "--cwd",
        default=".",
        help="Workspace root (default: current directory)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force the offline mock model even if an API key is configured",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Approve writes and commands without prompting",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one prompt and exit (old headless mode)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="Resume the latest session, or a session id",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full model reply instead of printing tokens as they arrive",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    cwd = str(Path(args.cwd).resolve())
    tools = create_default_tool_registry()
    mcp = connect_mcp(cwd)
    for tool in mcp.tools:
        tools.add(tool)
    if mcp.statuses:
        print(f"mcp:\n{mcp.format_status()}", file=sys.stderr)

    model_config = load_model_config()
    if args.mock or not model_config.is_configured:
        model = MockModelAdapter()
        print("config: using mock model", file=sys.stderr)
    else:
        protocol = resolve_protocol(model_config)
        model = create_model_adapter(model_config, tools, stream=not args.no_stream)
        print(
            f"config: protocol={protocol} model={model_config.model} "
            f"base_url={model_config.base_url} stream={'off' if args.no_stream else 'on'}",
            file=sys.stderr,
        )

    if args.yes:
        print("approval: auto (--yes)", file=sys.stderr)
        on_approve = None
    else:
        print("approval: prompt for writes and commands", file=sys.stderr)
        on_approve = _ask_approval

    on_ask_user = None if args.once else _ask_user
    if on_ask_user is None:
        print("ask_user: disabled (--once)", file=sys.stderr)

    callbacks = {
        "on_tool_start": _print_tool_start,
        "on_tool_result": _print_tool_result,
        "on_write_preview": _print_write_preview,
        "on_approve": on_approve,
        "on_ask_user": on_ask_user,
        "on_compact": _print_compact,
        "on_assistant_message": lambda text: print(text),
        "on_text_delta": (lambda chunk: print(chunk, end="", flush=True))
        if not args.no_stream
        else None,
    }
    store = SessionStore(cwd)
    memory = ProjectMemory(cwd)
    permissions = PermissionStore.load(cwd)
    todos = TodoStore()
    jobs = JobStore()
    system = build_system_prompt(SYSTEM_PROMPT, cwd)
    notes = memory.read()
    if notes:
        print(f"memory: loaded {len(notes)} chars from {memory.path}", file=sys.stderr)
    if permissions.writes or permissions.commands:
        print(
            f"permissions: {len(permissions.writes)} writes, "
            f"{len(permissions.commands)} commands from {permissions.path}",
            file=sys.stderr,
        )
    skill_count = len(discover_skills(cwd))
    if skill_count:
        print(f"skills: {skill_count} in {cwd}/.microcode/skills", file=sys.stderr)

    try:
        return _run_session(
            args,
            cwd,
            tools,
            model,
            callbacks,
            store,
            memory,
            system,
            mcp,
            permissions,
            todos,
            jobs,
        )
    finally:
        jobs.kill_all()
        mcp.close()


def _run_session(
    args, cwd, tools, model, callbacks, store, memory, system, mcp, permissions, todos, jobs
) -> int:
    if args.resume is not None:
        try:
            session = store.load(args.resume or None)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"session: resumed {session.id}", file=sys.stderr)
        messages = apply_system_message(list(session.messages), system)
        session.messages = messages
    else:
        messages = [{"role": "system", "content": system}]
        session = store.create(cwd, messages)
        store.save(session)
        print(f"session: {session.id}", file=sys.stderr)

    if args.once and not args.prompt:
        print("error: --once requires a prompt", file=sys.stderr)
        return 2

    if args.prompt:
        messages = run_user_turn(
            model=model,
            tools=tools,
            messages=messages,
            user_text=args.prompt,
            cwd=cwd,
            session=session,
            store=store,
            permissions=permissions,
            todos=todos,
            jobs=jobs,
            **callbacks,
        )

    if args.once:
        last = messages[-1]
        return 1 if last.get("role") != "assistant" else 0

    messages = run_repl(
        model=model,
        tools=tools,
        messages=messages,
        cwd=cwd,
        session=session,
        store=store,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        mcp_status=mcp.format_status(),
        permissions=permissions,
        todos=todos,
        jobs=jobs,
        **callbacks,
    )
    last = messages[-1]
    return 1 if last.get("role") != "assistant" else 0


if __name__ == "__main__":
    raise SystemExit(main())
