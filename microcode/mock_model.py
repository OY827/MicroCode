from __future__ import annotations

from microcode.types import AgentStep, ChatMessage


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _last_tool_result(messages: list[ChatMessage]) -> ChatMessage | None:
    for message in reversed(messages):
        if message.get("role") == "tool_result":
            return message
    return None


def _last_tool_name(messages: list[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant_tool_call":
            return str(message.get("toolName"))
    return None


class MockModelAdapter:
    """A fake model that follows a tiny script instead of calling an API.

    Real adapters will parse LLM tool-call JSON. This one uses slash commands
    and a default explore path so the loop can be tested offline.
    """

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        tool_result = _last_tool_result(messages)
        if tool_result is not None:
            last_call = _last_tool_name(messages)
            content = str(tool_result.get("content", ""))
            if last_call in {"list_files", "file_tree"}:
                return AgentStep(
                    type="assistant",
                    content=f"Workspace contents:\n\n{content}",
                )
            if last_call == "read_file":
                return AgentStep(
                    type="assistant",
                    content=f"File contents:\n\n{content}",
                )
            if last_call in {"explore", "task"}:
                return AgentStep(
                    type="assistant",
                    content=f"Explore summary:\n\n{content}",
                )
            return AgentStep(
                type="assistant",
                content=f"Tool result:\n\n{content}",
            )

        user_text = _last_user_text(messages)

        if user_text.startswith("/cmd "):
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "run_command",
                        "input": {"command": user_text[len("/cmd ") :].strip()},
                    }
                ],
            )

        if user_text.startswith("/grep "):
            payload = user_text[len("/grep ") :].strip()
            pattern, _, search_path = payload.partition("::")
            tool_input: dict = {"pattern": pattern.strip()}
            if search_path.strip():
                tool_input["path"] = search_path.strip()
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "grep_files", "input": tool_input}],
            )

        if user_text.startswith("/write "):
            payload = user_text[len("/write ") :]
            path, separator, content = payload.partition("::")
            if not separator:
                return AgentStep(type="assistant", content="Usage: /write <path>::<content>")
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "write_file",
                        "input": {"path": path.strip(), "content": content},
                    }
                ],
            )

        if user_text.startswith("/edit "):
            parts = user_text[len("/edit ") :].split("::")
            if len(parts) != 3:
                return AgentStep(
                    type="assistant",
                    content="Usage: /edit <path>::<search>::<replace>",
                )
            path, search, replace = parts
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "edit_file",
                        "input": {
                            "path": path.strip(),
                            "search": search,
                            "replace": replace,
                        },
                    }
                ],
            )

        if user_text.startswith("/patch "):
            payload = user_text[len("/patch ") :]
            path, separator, rest = payload.partition("::")
            hunks = [part for part in rest.split("::") if part]
            replacements = []
            for hunk in hunks:
                search, arrow, replace = hunk.partition("=>")
                if not arrow:
                    return AgentStep(
                        type="assistant",
                        content="Usage: /patch <path>::<search>=><replace>[::<search>=><replace>...]",
                    )
                replacements.append({"search": search, "replace": replace})
            if not separator or not replacements:
                return AgentStep(
                    type="assistant",
                    content="Usage: /patch <path>::<search>=><replace>[::<search>=><replace>...]",
                )
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "patch_file",
                        "input": {"path": path.strip(), "replacements": replacements},
                    }
                ],
            )

        if user_text.startswith("/read "):
            path = user_text[len("/read ") :].strip()
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "read_file", "input": {"path": path}}],
            )

        if user_text.startswith("/git "):
            payload = user_text[len("/git ") :].strip()
            action, separator, message = payload.partition(" ")
            if action not in {"status", "diff", "log", "add", "commit"}:
                return AgentStep(type="assistant", content="Usage: /git <status|diff|log|add|commit> [args]")
            tool_input: dict = {"action": action}
            if action == "commit":
                if not message.strip():
                    return AgentStep(type="assistant", content="Usage: /git commit <message>")
                tool_input["message"] = message.strip()
            elif action == "add":
                paths = message.split()
                if not paths:
                    return AgentStep(type="assistant", content="Usage: /git add <path> [path...]")
                tool_input["paths"] = paths
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "git", "input": tool_input}],
            )

        if user_text.startswith("/explore "):
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "explore",
                        "input": {"task": user_text[len("/explore ") :].strip()},
                    }
                ],
            )

        if user_text.startswith("/task "):
            payload = user_text[len("/task ") :].strip()
            agent_type, separator, prompt = payload.partition(" ")
            if agent_type not in {"explore", "plan", "general"} or not prompt.strip():
                return AgentStep(
                    type="assistant",
                    content="Usage: /task <explore|plan|general> <prompt>",
                )
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "task",
                        "input": {"description": prompt.strip(), "prompt": prompt.strip(), "agent_type": agent_type},
                    }
                ],
            )

        if user_text.startswith("/search "):
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "web_search",
                        "input": {"query": user_text[len("/search ") :].strip()},
                    }
                ],
            )

        if user_text.startswith("/symbols"):
            payload = user_text[len("/symbols") :].strip()
            tool_input: dict = {"name": payload} if payload else {}
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "find_symbols", "input": tool_input}],
            )

        if user_text.startswith("/refs "):
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "find_references",
                        "input": {"name": user_text[len("/refs ") :].strip()},
                    }
                ],
            )

        if user_text == "/review" or user_text.startswith("/review "):
            path = user_text[len("/review") :].strip()
            tool_input: dict = {"path": path} if path else {}
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "code_review", "input": tool_input}],
            )

        if user_text == "/test" or user_text.startswith("/test "):
            path = user_text[len("/test") :].strip()
            tool_input: dict = {"path": path} if path else {}
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "test_runner", "input": tool_input}],
            )

        if user_text.startswith("/copy "):
            payload = user_text[len("/copy ") :].strip()
            source, separator, destination = payload.partition("::")
            if not separator or not source.strip() or not destination.strip():
                return AgentStep(type="assistant", content="Usage: /copy <source>::<destination>")
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "batch_copy",
                        "input": {"source": source.strip(), "destination": destination.strip()},
                    }
                ],
            )

        if user_text.startswith("/move "):
            payload = user_text[len("/move ") :].strip()
            source, separator, destination = payload.partition("::")
            if not separator or not source.strip() or not destination.strip():
                return AgentStep(type="assistant", content="Usage: /move <source>::<destination>")
            return AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "mock-1",
                        "toolName": "batch_move",
                        "input": {"source": source.strip(), "destination": destination.strip()},
                    }
                ],
            )

        if user_text == "/delete" or user_text.startswith("/delete "):
            payload = user_text[len("/delete") :].strip()
            path, separator, flag = payload.partition("::")
            if not path.strip():
                return AgentStep(type="assistant", content="Usage: /delete <path>[::recursive]")
            tool_input: dict = {"path": path.strip()}
            if separator:
                if flag.strip().lower() not in {"recursive", "true", "1"}:
                    return AgentStep(type="assistant", content="Usage: /delete <path>[::recursive]")
                tool_input["recursive"] = True
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "batch_delete", "input": tool_input}],
            )

        if user_text.startswith("/tree"):
            payload = user_text[len("/tree") :].strip()
            path, separator, depth = payload.partition("::")
            tool_input: dict = {}
            if path.strip():
                tool_input["path"] = path.strip()
            if separator and depth.strip().isdigit():
                tool_input["max_depth"] = int(depth.strip())
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "file_tree", "input": tool_input}],
            )

        if user_text.startswith("/ls"):
            directory = user_text[len("/ls") :].strip()
            payload: dict = {"path": directory} if directory else {}
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "list_files", "input": payload}],
            )

        return AgentStep(
            type="tool_calls",
            calls=[{"id": "mock-1", "toolName": "list_files", "input": {}}],
        )
