from pathlib import Path

from microcode.compact import COMPACT_PREFIX, compact_messages, context_chars
from microcode.repl import run_repl, run_user_turn
from microcode.session import SessionStore, summarize_session
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.seen.append(list(messages))
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                validator=lambda value: value,
                run=lambda input_data, _context: ToolResult(ok=True, output=input_data["text"]),
            )
        ]
    )


def _history_with_old_tool_result(blob: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old task"},
        {
            "role": "assistant_tool_call",
            "toolUseId": "1",
            "toolName": "read_file",
            "input": {"path": "big.txt"},
        },
        {
            "role": "tool_result",
            "toolUseId": "1",
            "toolName": "read_file",
            "content": blob,
        },
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new task"},
    ]


def test_compact_shrinks_old_tool_result_keeps_system_and_recent() -> None:
    blob = "x" * 2000
    messages, report = compact_messages(
        _history_with_old_tool_result(blob),
        max_chars=500,
    )
    assert report.compacted >= 1
    assert report.after < report.before
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[-1]["content"] == "new task"
    assert str(messages[3]["content"]).startswith(COMPACT_PREFIX)
    assert "2000 chars" in str(messages[3]["content"])
    assert str(messages[3]["content"]) != blob


def test_under_threshold_is_noop() -> None:
    messages = _history_with_old_tool_result("tiny")
    compacted, report = compact_messages(messages, max_chars=80_000)
    assert report.compacted == 0
    assert compacted[3]["content"] == "tiny"


def test_force_compacts_even_under_threshold() -> None:
    blob = "y" * 2000
    compacted, report = compact_messages(
        _history_with_old_tool_result(blob),
        max_chars=80_000,
        force=True,
    )
    assert report.compacted >= 1
    assert str(compacted[3]["content"]).startswith(COMPACT_PREFIX)


def test_already_compacted_is_not_double_wrapped() -> None:
    messages = _history_with_old_tool_result("z" * 2000)
    once, _ = compact_messages(messages, force=True)
    twice, report = compact_messages(once, force=True)
    assert report.compacted == 0
    assert twice[3]["content"] == once[3]["content"]


def test_recent_tool_result_is_kept() -> None:
    blob = "w" * 2000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "now"},
        {
            "role": "tool_result",
            "toolUseId": "1",
            "toolName": "read_file",
            "content": blob,
        },
    ]
    compacted, report = compact_messages(messages, max_chars=500)
    assert report.compacted == 0
    assert compacted[2]["content"] == blob


def test_large_tool_call_input_is_compacted() -> None:
    payload = {"path": "a.py", "content": "print(1)\n" * 400}
    messages, report = compact_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {
                "role": "assistant_tool_call",
                "toolUseId": "1",
                "toolName": "write_file",
                "input": payload,
            },
            {"role": "user", "content": "next"},
        ],
        force=True,
    )
    assert report.compacted == 1
    compacted_input = messages[2]["input"]
    assert compacted_input["_compacted"] is True
    assert compacted_input["path"] == "a.py"
    assert compacted_input["original_chars"] > 400


def test_run_user_turn_compacts_before_model_sees_history() -> None:
    blob = "q" * 2000
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {
            "role": "tool_result",
            "toolUseId": "1",
            "toolName": "read_file",
            "content": blob,
        },
        {"role": "assistant", "content": "done"},
    ]
    model = ScriptedModel([AgentStep(type="assistant", content="ok")])
    reports: list[str] = []
    run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=history,
        user_text="next",
        cwd=".",
        compact_max_chars=800,
        on_compact=lambda report: reports.append(report.summary()),
    )
    seen = model.seen[0]
    old_result = next(message for message in seen if message.get("role") == "tool_result")
    assert str(old_result["content"]).startswith(COMPACT_PREFIX)
    assert reports and "compacted" in reports[0]


def test_repl_compact_command_persists(tmp_path: Path) -> None:
    blob = "n" * 2000
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), _history_with_old_tool_result(blob))
    store.save(session)
    lines = iter(["/compact", "/exit"])

    messages = run_repl(
        model=ScriptedModel([]),
        tools=_echo_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        read_line=lambda: next(lines) + "\n",
    )

    assert str(messages[3]["content"]).startswith(COMPACT_PREFIX)
    loaded = store.load(session.id)
    assert str(loaded.messages[3]["content"]).startswith(COMPACT_PREFIX)
    assert loaded.checkpoint == []
    assert "chars:" in summarize_session(loaded)
    assert context_chars(loaded.messages) < 2000 + 100
