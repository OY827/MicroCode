from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    hint: str
    usage: str


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "Show local commands", "/help"),
    SlashCommand("/status", "Snapshot of this chat", "/status"),
    SlashCommand("/replay", "Replay this chat or a saved id", "/replay"),
    SlashCommand("/tape", "Show the turn tape (model and tools)", "/tape "),
    SlashCommand("/fixture", "Save this chat as an offline pytest replay", "/fixture"),
    SlashCommand("/model", "Show or switch the model", "/model "),
    SlashCommand("/cost", "Tokens and estimated spend", "/cost"),
    SlashCommand("/session", "Show the current saved session", "/session"),
    SlashCommand("/sessions", "List recent saved sessions", "/sessions"),
    SlashCommand("/memory", "Show or edit project memory", "/memory "),
    SlashCommand("/skills", "List local SKILL.md files", "/skills"),
    SlashCommand("/mcp", "Show configured MCP servers", "/mcp"),
    SlashCommand("/permissions", "Show remembered allow rules", "/permissions"),
    SlashCommand("/todos", "Show the in-memory task list", "/todos"),
    SlashCommand("/jobs", "Show background shell jobs", "/jobs"),
    SlashCommand("/readiness", "Local checkup, no network", "/readiness"),
    SlashCommand("/init", "Scan the repo for how to test", "/init"),
    SlashCommand("/compact", "Shrink older tool results", "/compact"),
    SlashCommand("/undo", "Restore the last write group", "/undo"),
    SlashCommand("/redo", "Re-apply the last undo or rewind", "/redo"),
    SlashCommand("/checkpoints", "List rewind checkpoints", "/checkpoints"),
    SlashCommand("/rewind-preview", "Show what a rewind would restore", "/rewind-preview "),
    SlashCommand("/rewind", "Restore files from a checkpoint", "/rewind "),
    SlashCommand("/mode", "Show or set permission mode", "/mode "),
    SlashCommand("/collapse", "Collapse expanded tool output", "/collapse"),
    SlashCommand("/ls", "List one folder (no model)", "/ls "),
    SlashCommand("/tree", "Show a directory tree (no model)", "/tree "),
    SlashCommand("/grep", "Search files (no model)", "/grep "),
    SlashCommand("/read", "Read a file (no model)", "/read "),
    SlashCommand("/write", "Create or overwrite a file", "/write "),
    SlashCommand("/edit", "Replace one unique string", "/edit "),
    SlashCommand("/patch", "Several replacements in one file", "/patch "),
    SlashCommand("/copy", "Copy a file or folder", "/copy "),
    SlashCommand("/move", "Rename or move", "/move "),
    SlashCommand("/delete", "Delete a file or folder", "/delete "),
    SlashCommand("/cmd", "Run a shell command (no model)", "/cmd "),
    SlashCommand("/git", "Inspect or commit git", "/git "),
    SlashCommand("/exit", "Leave the session", "/exit"),
    SlashCommand("/quit", "Same as /exit", "/quit"),
)


def matching_slash_commands(text: str) -> list[SlashCommand]:
    stripped = text.strip()
    if not stripped.startswith("/") or " " in stripped:
        return []
    prefix = stripped.lower()
    return [command for command in SLASH_COMMANDS if command.name.startswith(prefix)]
