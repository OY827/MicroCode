"""Run one offline turn and print the turn tape.

Usage (from the repo root):

    python scripts/demo_turn_tape.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from microcode.agent_loop import run_agent_turn
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.turn_tape import TurnTape, format_tape, tape_path
from microcode.types import AgentStep


class ScriptedModel:
    def __init__(self) -> None:
        self.calls = 0

    def next(self, messages, on_text_delta=None) -> AgentStep:
        self.calls += 1
        if self.calls == 1:
            return AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "echo", "input": {"text": "hello tape"}}],
            )
        return AgentStep(type="assistant", content="echoed hello tape")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microcode-tape-") as folder:
        root = Path(folder)
        tape = TurnTape(root, session_id="demo")
        tape.record_user("say hello through the echo tool")
        tools = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="echo",
                    input_schema={"type": "object"},
                    validator=lambda value: value,
                    run=lambda data, _context: ToolResult(ok=True, output=f"echo:{data['text']}"),
                )
            ]
        )
        run_agent_turn(
            model=ScriptedModel(),
            tools=tools,
            messages=[
                {"role": "system", "content": "demo"},
                {"role": "user", "content": "say hello through the echo tool"},
            ],
            cwd=str(root),
            tape=tape,
        )
        path = tape_path(root)
        print(path.read_text(encoding="utf-8"), end="")
        print("---")
        print(format_tape(root), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
