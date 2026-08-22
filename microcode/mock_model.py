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

    def next(self, messages: list[ChatMessage]) -> AgentStep:
        tool_result = _last_tool_result(messages)
        if tool_result is not None:
            last_call = _last_tool_name(messages)
            content = str(tool_result.get("content", ""))
            if last_call == "list_files":
                return AgentStep(
                    type="assistant",
                    content=f"Workspace contents:\n\n{content}",
                )
            if last_call == "read_file":
                return AgentStep(
                    type="assistant",
                    content=f"File contents:\n\n{content}",
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

        if user_text.startswith("/read "):
            path = user_text[len("/read ") :].strip()
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "mock-1", "toolName": "read_file", "input": {"path": path}}],
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
