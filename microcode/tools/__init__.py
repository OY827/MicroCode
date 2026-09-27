from __future__ import annotations

from microcode.tooling import ToolDefinition, ToolRegistry
from microcode.tools.ask_user import ask_user_tool
from microcode.tools.await_job import await_job_tool
from microcode.tools.batch_ops import batch_copy_tool, batch_delete_tool, batch_move_tool
from microcode.tools.code_nav import find_references_tool, find_symbols_tool
from microcode.tools.code_review import code_review_tool
from microcode.tools.edit_file import edit_file_tool
from microcode.tools.explore import explore_tool
from microcode.tools.file_tree import file_tree_tool
from microcode.tools.git import git_tool
from microcode.tools.grep_files import grep_files_tool
from microcode.tools.list_files import list_files_tool
from microcode.tools.load_skill import load_skill_tool
from microcode.tools.patch_file import patch_file_tool
from microcode.tools.read_file import read_file_tool
from microcode.tools.run_command import run_command_tool
from microcode.tools.task import NESTED_AGENT_TOOLS, task_tool
from microcode.tools.test_runner import test_runner_tool
from microcode.tools.todo_write import todo_write_tool
from microcode.tools.web_fetch import web_fetch_tool
from microcode.tools.web_search import web_search_tool
from microcode.tools.write_file import write_file_tool


def _builtin_tools() -> list[ToolDefinition]:
    return [
        list_files_tool,
        file_tree_tool,
        read_file_tool,
        grep_files_tool,
        find_symbols_tool,
        find_references_tool,
        code_review_tool,
        edit_file_tool,
        patch_file_tool,
        write_file_tool,
        batch_copy_tool,
        batch_move_tool,
        batch_delete_tool,
        git_tool,
        run_command_tool,
        test_runner_tool,
        await_job_tool,
        todo_write_tool,
        ask_user_tool,
        web_fetch_tool,
        web_search_tool,
        load_skill_tool,
        explore_tool,
        task_tool,
    ]


def create_default_tool_registry() -> ToolRegistry:
    return ToolRegistry(_builtin_tools())


def create_subagent_tool_registry(allowed: frozenset[str] | None) -> ToolRegistry:
    """Tools for a nested agent. task/explore are never included, so it cannot recurse."""

    tools = [tool for tool in _builtin_tools() if tool.name not in NESTED_AGENT_TOOLS]
    if allowed is not None:
        tools = [tool for tool in tools if tool.name in allowed]
    return ToolRegistry(tools)


def create_explore_tool_registry() -> ToolRegistry:
    return create_subagent_tool_registry(
        frozenset({"list_files", "file_tree", "read_file", "grep_files", "find_symbols", "find_references"})
    )
