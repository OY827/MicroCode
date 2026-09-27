from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcode.config import load_dotenv, load_model_config, resolve_protocol
from microcode.readiness import build_readiness_report
from microcode.mcp import connect_mcp
from microcode.memory import ProjectMemory, apply_system_message
from microcode.memory_reranker import MemoryReranker
from microcode.mock_model import MockModelAdapter
from microcode.models import apply_model_switch, create_model_adapter, model_label
from microcode.permissions import PermissionStore
from microcode.prompt import build_system_prompt
from microcode.repl import run_repl, run_user_turn
from microcode.session import SessionStore
from microcode.session_note import finish_session_memory
from microcode.skills import discover_skills
from microcode.todos import TodoStore
from microcode.jobs import JobStore
from microcode.tools import create_default_tool_registry
from microcode.tui import TuiContext, run_tui, should_use_tui
from microcode.usage import UsageLedger

SYSTEM_PROMPT = (
    "You are MicroCode, a local coding agent working in a real repository. "
    "Use list_files for one folder, file_tree for limited-depth structure, "
    "find_symbols/find_references for Python definitions and uses, "
    "and grep_files or read_file before changing anything. "
    "Use edit_file for one unique search/replace, patch_file for several replacements in one file, "
    "and write_file to create or overwrite files. "
    "Use batch_copy, batch_move, and batch_delete to copy, rename, or remove files; do not use run_command for that. "
    "Use git for status, diff, log, add, and commit. "
    "Use test_runner to discover and run pytest/unittest after edits. "
    "Use code_review for a static pass on unused imports, empty excepts, and long functions. "
    "Use run_command for other scripts or git verbs such as push. "
    "For long-running commands, set background=true and later collect output with await_job. "
    "If local skills are listed below, use load_skill to load the matching SKILL.md before following it. "
    "For tasks with more than two steps, use todo_write to keep a short checklist and update it as you go. "
    "If a key choice is unspecified and guessing would change the code, use ask_user before editing. "
    "Use web_search to find public documentation URLs, then web_fetch to read one. "
    "Do not use web_fetch for local or private addresses. "
    "For broad codebase questions, use task with agent_type=explore so a read-only sub-agent searches and returns a summary. "
    "Use task with agent_type=plan before a large change, or agent_type=general to isolate a multi-step edit. "
    "If a compacted tool result was saved to a path, call read_file on that path to recover it. "
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


def _ask_revise_write(path: str, proposed: str, preview: str):
    """Approve a file write. `e` replaces the proposed text before it is saved."""

    from microcode.tooling import WriteDecision

    print(f"? Write {path}?", file=sys.stderr)
    if preview:
        print(preview, file=sys.stderr)
    print("Approve? [y/N/a/e] ", end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if not line:
        return WriteDecision(allow=False)
    text = line.strip().lower()
    if text in {"a", "always"}:
        return WriteDecision(allow=True, remember=True)
    if text in {"y", "yes"}:
        return WriteDecision(allow=True)
    if text != "e":
        return WriteDecision(allow=False)
    print(
        "Paste the file to write. Finish with a line that contains only .",
        file=sys.stderr,
    )
    body: list[str] = []
    while True:
        row = sys.stdin.readline()
        if not row or row.strip() == ".":
            break
        body.append(row.rstrip("\n"))
    content = "\n".join(body)
    if proposed.endswith("\n"):
        content += "\n"
    return WriteDecision(allow=True, content=content)


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
    tui_group = parser.add_mutually_exclusive_group()
    tui_group.add_argument(
        "--tui",
        action="store_true",
        help="Force the full-screen terminal UI",
    )
    tui_group.add_argument(
        "--no-tui",
        action="store_true",
        help="Use the line prompt instead of the full-screen UI",
    )
    parser.add_argument(
        "--readiness",
        action="store_true",
        help="Print a local config checkup and exit (does not call the model)",
    )
    parser.add_argument(
        "--readiness-json",
        action="store_true",
        help="Same checkup as --readiness, as JSON",
    )
    parser.add_argument(
        "--fail-on",
        choices=("ready", "warning", "blocked"),
        default="blocked",
        help="Exit code 2 when the checkup is this status or worse (default: blocked)",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    if args.readiness or args.readiness_json:
        report = build_readiness_report()
        if args.readiness_json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.format(), end="")
        return report.exit_code(args.fail_on)
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
        "on_revise_write": None if args.yes or args.once else _ask_revise_write,
        "on_ask_user": on_ask_user,
        "on_compact": _print_compact,
        "on_assistant_message": lambda text: print(text),
        "on_text_delta": (lambda chunk: print(chunk, end="", flush=True))
        if not args.no_stream
        else None,
    }
    store = SessionStore(cwd)
    memory = ProjectMemory(cwd, reranker=MemoryReranker.from_model(model))
    permissions = PermissionStore.load(cwd)
    todos = TodoStore()
    jobs = JobStore()
    system = build_system_prompt(SYSTEM_PROMPT, cwd, memory=memory)
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
        if session.model_name and session.model_name != model_label(model):
            model, switch_message = apply_model_switch(model, tools, session.model_name)
            print(f"session model: {switch_message}", file=sys.stderr)
    else:
        messages = [{"role": "system", "content": system}]
        session = store.create(cwd, messages)
        session.model_name = model_label(model)
        store.save(session)
        print(f"session: {session.id}", file=sys.stderr)

    usage = UsageLedger.from_dict(session.usage)
    if args.yes:
        session.permission_mode = "yes"
        store.save(session)
    if session.permission_mode != "ask":
        print(f"permission mode: {session.permission_mode}", file=sys.stderr)

    if args.once and not args.prompt:
        print("error: --once requires a prompt", file=sys.stderr)
        return 2

    force_tui = True if args.tui else False if args.no_tui else None
    use_tui = should_use_tui(once=args.once, force=force_tui)

    if args.prompt and not use_tui:
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
            system_prompt=SYSTEM_PROMPT,
            memory=memory,
            usage=usage,
            **callbacks,
        )

    if args.once:
        print(finish_session_memory(memory, messages=messages, session=session), file=sys.stderr)
        last = messages[-1]
        return 1 if last.get("role") != "assistant" else 0

    if use_tui:
        ctx = TuiContext(
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
            usage=usage,
        )
        messages = run_tui(ctx=ctx, initial_prompt=args.prompt)
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
        usage=usage,
        **callbacks,
    )
    last = messages[-1]
    return 1 if last.get("role") != "assistant" else 0


if __name__ == "__main__":
    raise SystemExit(main())
