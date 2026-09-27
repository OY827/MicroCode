from microcode.agent_loop import run_agent_turn
from microcode.anthropic_adapter import consume_anthropic_stream, parse_anthropic_response
from microcode.openai_adapter import consume_openai_stream, parse_openai_response
from microcode.session import SessionStore
from microcode.tooling import ToolDefinition, ToolRegistry, ToolResult
from microcode.types import AgentStep, ChatMessage, TokenUsage
from microcode.usage import UsageLedger, estimate_usd, usage_from_openai


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _echo() -> ToolRegistry:
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


def test_ledger_uses_api_usage_when_present() -> None:
    ledger = UsageLedger()
    ledger.record(
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=1000, output_tokens=200),
        messages=[{"role": "user", "content": "hi"}],
        step=AgentStep(type="assistant", content="ok"),
    )
    assert ledger.calls == 1
    assert ledger.input_tokens == 1000
    assert ledger.output_tokens == 200
    assert ledger.estimated_calls == 0
    text = ledger.format(model_name="gpt-4o-mini")
    assert "model calls: 1" in text
    assert "estimated cost:" in text
    assert "gpt-4o-mini" in text


def test_ledger_estimates_when_api_omits_usage() -> None:
    ledger = UsageLedger()
    messages = [{"role": "user", "content": "x" * 40}]
    ledger.record(
        model="mock",
        usage=None,
        messages=messages,
        step=AgentStep(type="assistant", content="y" * 8),
    )
    assert ledger.calls == 1
    assert ledger.estimated_calls == 1
    assert ledger.input_tokens >= 1
    text = ledger.format(model_name="mock")
    assert "not a real bill" in text
    assert "characters/4" in text


def test_price_table_matches_known_models() -> None:
    assert estimate_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert estimate_usd("claude-sonnet-4", 0, 1_000_000) == 15.0


def test_parse_openai_keeps_usage() -> None:
    step = parse_openai_response(
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )
    assert step.content == "hello"
    assert step.usage is not None
    assert step.usage.input_tokens == 11
    assert step.usage.output_tokens == 4


def test_consume_openai_stream_reads_final_usage_chunk() -> None:
    step = consume_openai_stream(
        [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
        ]
    )
    assert step.content == "Hi"
    assert step.usage is not None
    assert step.usage.input_tokens == 9
    assert step.usage.output_tokens == 2


def test_parse_anthropic_keeps_usage() -> None:
    step = parse_anthropic_response(
        {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    )
    assert step.usage is not None
    assert step.usage.input_tokens == 7
    assert step.usage.output_tokens == 3


def test_consume_anthropic_stream_merges_usage() -> None:
    step = consume_anthropic_stream(
        [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 12, "output_tokens": 0}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
            {"type": "message_delta", "usage": {"output_tokens": 5}},
        ]
    )
    assert step.content == "Hi"
    assert step.usage is not None
    assert step.usage.input_tokens == 12
    assert step.usage.output_tokens == 5


def test_agent_loop_records_usage() -> None:
    ledger = UsageLedger()
    run_agent_turn(
        model=ScriptedModel(
            [AgentStep(type="assistant", content="done", usage=TokenUsage(8, 2))]
        ),
        tools=_echo(),
        messages=[{"role": "user", "content": "hi"}],
        cwd=".",
        usage=ledger,
    )
    assert ledger.calls == 1
    assert ledger.input_tokens == 8
    assert ledger.output_tokens == 2


def test_session_roundtrip_keeps_usage(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "user", "content": "hi"}])
    session.model_name = "gpt-4o-mini"
    session.usage = UsageLedger(calls=2, input_tokens=30, output_tokens=10).to_dict()
    store.save(session)
    loaded = store.load(session.id)
    assert loaded.model_name == "gpt-4o-mini"
    assert loaded.usage["calls"] == 2
    assert usage_from_openai({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}) is not None
