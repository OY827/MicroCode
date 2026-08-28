from __future__ import annotations

from microcode.tooling import ToolRegistry
from microcode.tools.ask_user import ask_user_tool
from microcode.tools.await_job import await_job_tool
from microcode.tools.edit_file import edit_file_tool
from microcode.tools.explore import explore_tool
from microcode.tools.grep_files import grep_files_tool
from microcode.tools.list_files import list_files_tool
from microcode.tools.load_skill import load_skill_tool
from microcode.tools.read_file import read_file_tool
from microcode.tools.run_command import run_command_tool
from microcode.tools.todo_write import todo_write_tool
from microcode.tools.web_fetch import web_fetch_tool
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
            await_job_tool,
            todo_write_tool,
            ask_user_tool,
            web_fetch_tool,
            load_skill_tool,
            explore_tool,
        ]
    )


def create_explore_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            list_files_tool,
            read_file_tool,
            grep_files_tool,
        ]
    )
