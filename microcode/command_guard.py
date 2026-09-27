from __future__ import annotations

import os
import shlex
from pathlib import Path

CONTROL_TOKENS = {"|", "||", "&&", ";", "&", ">", "<", ">>"}


def _strip_wrapping_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def split_command_line(command: str) -> list[str]:
    """Split one command into argv. Does not go through a system shell."""

    try:
        return shlex.split(command, posix=True)
    except ValueError:
        if os.name != "nt":
            raise
        try:
            return [_strip_wrapping_quotes(token) for token in shlex.split(command, posix=False)]
        except ValueError:
            return command.split()


def _program_name(token: str) -> str:
    name = Path(str(token).replace("\\", "/")).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    if name.endswith(".cmd") or name.endswith(".bat"):
        name = name.rsplit(".", 1)[0]
    return name


def _flags(args: list[str]) -> str:
    return "".join(arg[1:] for arg in args if arg.startswith("-")).lower()


def refuse_command(command: str) -> str | None:
    """Why this line must not run. None means it may proceed (still asks the user)."""

    raw = (command or "").strip()
    if not raw:
        return "Command is empty."
    if "\n" in raw or "\r" in raw:
        return "Refusing a multi-line command."
    try:
        tokens = split_command_line(raw)
    except ValueError as error:
        return f"Could not split command: {error}"
    if not tokens:
        return "Command is empty."
    for token in tokens:
        if token in CONTROL_TOKENS or token.startswith(">") or token.startswith("<"):
            return (
                "Refusing shell chaining or redirection. "
                "Run one program at a time (no |, &&, ;, or >)."
            )
    program = _program_name(tokens[0])
    args = tokens[1:]
    lowered = [arg.lower() for arg in args]
    reason = _dangerous(program, args, lowered)
    if reason:
        return f"Refusing command: {reason}"
    return None


def _dangerous(program: str, args: list[str], lowered: list[str]) -> str | None:
    if program in {"format", "mkfs", "fdisk", "diskpart", "dd", "shutdown", "reboot"}:
        return f"{program} can destroy the machine or disk"
    if program == "rm":
        flags = _flags(args)
        if "r" in flags and "f" in flags:
            return "rm -rf"
    if program in {"del", "erase"} and any(arg in {"/s", "/s/q", "/q/s"} for arg in lowered):
        return "recursive Windows delete"
    if program in {"rd", "rmdir"} and "/s" in lowered:
        return "recursive Windows rmdir"
    if program == "chmod" and any(arg.endswith("777") for arg in lowered):
        return "chmod 777"
    if program == "git":
        if "reset" in lowered and "--hard" in lowered:
            return "git reset --hard"
        if "clean" in lowered:
            return "git clean"
        if "push" in lowered and any(arg in {"--force", "-f"} for arg in lowered):
            return "git push --force"
    if program in {"powershell", "pwsh"} and any(
        arg in {"-enc", "-encodedcommand", "iex", "invoke-expression"} for arg in lowered
    ):
        return "PowerShell encoded or invoke-expression"
    return None


def argv_for(command: str) -> list[str]:
    """Split after refuse_command returned None."""

    return split_command_line(command.strip())
