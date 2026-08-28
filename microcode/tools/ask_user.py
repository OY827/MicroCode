from __future__ import annotations

from microcode.tooling import ToolDefinition, ToolResult

NO_USER_MESSAGE = (
    "No user available to answer. Do not guess; state assumptions or stop."
)


def _validate(input_data: dict) -> dict:
    question = input_data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    return {"question": question.strip()}


def _run(input_data: dict, context) -> ToolResult:
    answer = context.ask(input_data["question"])
    if answer is None:
        return ToolResult(ok=False, output=NO_USER_MESSAGE)
    text = str(answer).strip()
    if not text:
        return ToolResult(ok=False, output="User did not answer.")
    return ToolResult(ok=True, output=text)


ask_user_tool = ToolDefinition(
    name="ask_user",
    description=(
        "Ask the human a clarifying question and wait for one line. "
        "Use this when a key choice is unspecified and guessing would change the code. "
        "Do not use it for write/command approval."
    ),
    input_schema={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
    validator=_validate,
    run=_run,
)
