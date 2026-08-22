from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcode.config import load_dotenv, load_model_config, resolve_protocol
from microcode.mock_model import MockModelAdapter
from microcode.models import create_model_adapter
from microcode.repl import run_repl, run_user_turn
from microcode.tools import create_default_tool_registry

SYSTEM_PROMPT = (
    "You are MicroCode, a local coding agent working in a real repository. "
    "Use list_files, grep_files, and read_file to inspect the workspace before changing it. "
    "Use edit_file for unique search/replace edits and write_file to create or overwrite files. "
    "Use run_command to execute tests or scripts in the workspace after making changes. "
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


def _ask_approval(summary: str) -> bool:
    print(f"? {summary}", file=sys.stderr)
    print("Approve? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return False
    if not line:
        return False
    return line.strip().lower() in {"y", "yes"}


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
    args = parser.parse_args(argv)
    load_dotenv()
    model_config = load_model_config()
    tools = create_default_tool_registry()
    if args.mock or not model_config.is_configured:
        model = MockModelAdapter()
        print("config: using mock model", file=sys.stderr)
    else:
        protocol = resolve_protocol(model_config)
        model = create_model_adapter(model_config, tools)
        print(
            f"config: protocol={protocol} model={model_config.model} "
            f"base_url={model_config.base_url}",
            file=sys.stderr,
        )

    cwd = str(Path(args.cwd).resolve())
    if args.yes:
        print("approval: auto (--yes)", file=sys.stderr)
        on_approve = None
    else:
        print("approval: prompt for writes and commands", file=sys.stderr)
        on_approve = _ask_approval

    callbacks = {
        "on_tool_start": _print_tool_start,
        "on_tool_result": _print_tool_result,
        "on_write_preview": _print_write_preview,
        "on_approve": on_approve,
        "on_assistant_message": lambda text: print(text),
    }
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

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
        **callbacks,
    )
    last = messages[-1]
    return 1 if last.get("role") != "assistant" else 0


if __name__ == "__main__":
    raise SystemExit(main())
