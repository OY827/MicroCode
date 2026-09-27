from __future__ import annotations

from dataclasses import dataclass

from microcode.checkpoint import WriteCheckpoint, remember_write_group
from microcode.tooling import ToolContext, ToolRegistry, ToolResult
from microcode.turn_tape import open_turn_tape


@dataclass(frozen=True)
class ToolShortcut:
    tool_name: str
    input_data: dict


def parse_tool_shortcut(text: str) -> ToolShortcut | str | None:
    """Parse a slash tool shortcut.

    Returns a ToolShortcut, a usage string, or None if the line is not a shortcut.
    `/lsfoo` is not `/ls`.
    """

    raw = text.strip()
    command = raw.lower()
    if command == "/ls" or command.startswith("/ls "):
        directory = raw[3:].strip()
        return ToolShortcut("list_files", {"path": directory} if directory else {})

    if command == "/tree" or command.startswith("/tree "):
        payload = raw[5:].strip()
        path, separator, depth = payload.partition("::")
        tool_input: dict = {}
        if path.strip():
            tool_input["path"] = path.strip()
        if separator:
            if not depth.strip().isdigit():
                return "Usage: /tree [path][::depth]"
            tool_input["max_depth"] = int(depth.strip())
        return ToolShortcut("file_tree", tool_input)

    if command == "/grep" or command.startswith("/grep "):
        payload = raw[5:].strip()
        if not payload:
            return "Usage: /grep <pattern>[::path]"
        pattern, _, search_path = payload.partition("::")
        if not pattern.strip():
            return "Usage: /grep <pattern>[::path]"
        tool_input = {"pattern": pattern.strip()}
        if search_path.strip():
            tool_input["path"] = search_path.strip()
        return ToolShortcut("grep_files", tool_input)

    if command == "/read" or command.startswith("/read "):
        path = raw[5:].strip()
        if not path:
            return "Usage: /read <path>"
        return ToolShortcut("read_file", {"path": path})

    if command == "/write" or command.startswith("/write "):
        payload = raw[6:].lstrip()
        path, separator, content = payload.partition("::")
        if not separator or not path.strip():
            return "Usage: /write <path>::<content>"
        return ToolShortcut("write_file", {"path": path.strip(), "content": content})

    if command == "/edit" or command.startswith("/edit "):
        parts = raw[5:].lstrip().split("::")
        if len(parts) != 3 or not parts[0].strip() or not parts[1]:
            return "Usage: /edit <path>::<search>::<replace>"
        return ToolShortcut(
            "edit_file",
            {"path": parts[0].strip(), "search": parts[1], "replace": parts[2]},
        )

    if command == "/patch" or command.startswith("/patch "):
        payload = raw[6:].lstrip()
        path, separator, rest = payload.partition("::")
        hunks = [part for part in rest.split("::") if part]
        replacements = []
        for hunk in hunks:
            search, arrow, replace = hunk.partition("=>")
            if not arrow:
                return "Usage: /patch <path>::<search>=><replace>[::<search>=><replace>...]"
            replacements.append({"search": search, "replace": replace})
        if not separator or not path.strip() or not replacements:
            return "Usage: /patch <path>::<search>=><replace>[::<search>=><replace>...]"
        return ToolShortcut(
            "patch_file",
            {"path": path.strip(), "replacements": replacements},
        )

    if command == "/copy" or command.startswith("/copy "):
        source, separator, destination = raw[5:].strip().partition("::")
        if not separator or not source.strip() or not destination.strip():
            return "Usage: /copy <source>::<destination>"
        return ToolShortcut(
            "batch_copy",
            {"source": source.strip(), "destination": destination.strip()},
        )

    if command == "/move" or command.startswith("/move "):
        source, separator, destination = raw[5:].strip().partition("::")
        if not separator or not source.strip() or not destination.strip():
            return "Usage: /move <source>::<destination>"
        return ToolShortcut(
            "batch_move",
            {"source": source.strip(), "destination": destination.strip()},
        )

    if command == "/delete" or command.startswith("/delete "):
        payload = raw[7:].strip()
        path, separator, flag = payload.partition("::")
        if not path.strip():
            return "Usage: /delete <path>[::recursive]"
        tool_input: dict = {"path": path.strip()}
        if separator:
            if flag.strip().lower() not in {"recursive", "true", "1"}:
                return "Usage: /delete <path>[::recursive]"
            tool_input["recursive"] = True
        return ToolShortcut("batch_delete", tool_input)

    if command == "/git" or command.startswith("/git "):
        payload = raw[4:].strip()
        action, _, rest = payload.partition(" ")
        action = action.lower()
        if action not in {"status", "diff", "log", "add", "commit"}:
            return "Usage: /git <status|diff|log|add|commit> [args]"
        tool_input: dict = {"action": action}
        if action == "commit":
            if not rest.strip():
                return "Usage: /git commit <message>"
            tool_input["message"] = rest.strip()
        elif action == "add":
            paths = rest.split()
            if not paths:
                return "Usage: /git add <path> [path...]"
            tool_input["paths"] = paths
        return ToolShortcut("git", tool_input)

    if command == "/cmd" or command.startswith("/cmd "):
        payload = raw[4:].strip()
        cwd, separator, command_text = payload.partition("::")
        text = command_text.strip() if separator else payload
        workdir = cwd.strip() if separator else ""
        if not text:
            return "Usage: /cmd <command>  |  /cmd <cwd>::<command>"
        tool_input = {"command": text}
        if workdir:
            tool_input["cwd"] = workdir
        return ToolShortcut("run_command", tool_input)

    return None


def execute_tool_shortcut(
    shortcut: ToolShortcut,
    *,
    tools: ToolRegistry,
    cwd: str,
    on_write_preview=None,
    on_approve=None,
    on_revise_write=None,
    session=None,
    store=None,
    permissions=None,
    jobs=None,
) -> ToolResult:
    checkpoint = WriteCheckpoint()
    permission_mode = session.permission_mode if session is not None else "ask"
    result = tools.execute(
        shortcut.tool_name,
        shortcut.input_data,
        ToolContext(
            cwd=cwd,
            on_write_preview=on_write_preview,
            on_approve=on_approve,
            on_revise_write=on_revise_write,
            checkpoint=checkpoint,
            permissions=permissions,
            jobs=jobs,
            permission_mode=permission_mode,
        ),
    )
    tape = open_turn_tape(
        cwd,
        session.id if session is not None else "",
        via="shortcut",
    )
    if tape is not None:
        tape.record_tool(
            name=shortcut.tool_name,
            tool_input=shortcut.input_data,
            ok=result.ok,
            output=result.output,
        )
    if session is not None and checkpoint:
        remember_write_group(session, checkpoint)
        if store is not None:
            store.save(session)
    return result
