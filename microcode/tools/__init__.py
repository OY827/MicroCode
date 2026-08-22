from __future__ import annotations

from microcode.tooling import ToolRegistry
from microcode.tools.edit_file import edit_file_tool
from microcode.tools.grep_files import grep_files_tool
from microcode.tools.list_files import list_files_tool
from microcode.tools.read_file import read_file_tool
from microcode.tools.run_command import run_command_tool
from microcode.tools.write_file import write_file_tool


def create_default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            list_files_tool,
            read_file_tool,
            grep_files_tool,
            edit_file_tool,
            write_file_tool,
            run_command_tool,
        ]
    )
