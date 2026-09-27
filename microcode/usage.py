from __future__ import annotations

from dataclasses import dataclass, field

from microcode.compact import compact_limit, context_chars
from microcode.types import AgentStep, ChatMessage, TokenUsage

CHARS_PER_TOKEN = 4

# USD per 1M tokens: (input, output). First matching substring wins.
_PRICES: tuple[tuple[str, float, float], ...] = (
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4o", 2.50, 10.0),
    ("gpt-4", 10.0, 30.0),
    ("o3-mini", 1.10, 4.40),
    ("o1-mini", 3.0, 12.0),
    ("o3", 2.00, 8.00),
    ("o1", 15.0, 60.0),
    ("claude-haiku", 0.80, 4.0),
    ("claude-sonnet", 3.0, 15.0),
    ("claude-opus", 15.0, 75.0),
    ("deepseek", 0.14, 0.28),
    ("qwen", 0.22, 0.88),
)
_DEFAULT_PRICE = (3.0, 15.0)


def _as_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def usage_from_openai(data: dict) -> TokenUsage | None:
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    inp = _as_int(raw.get("prompt_tokens") if raw.get("prompt_tokens") is not None else raw.get("input_tokens"))
    out = _as_int(
        raw.get("completion_tokens") if raw.get("completion_tokens") is not None else raw.get("output_tokens")
    )
    if inp == 0 and out == 0:
        return None
    return TokenUsage(input_tokens=inp, output_tokens=out)


def usage_from_anthropic(data: dict) -> TokenUsage | None:
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    inp = _as_int(raw.get("input_tokens"))
    out = _as_int(raw.get("output_tokens"))
    if inp == 0 and out == 0:
        return None
    return TokenUsage(input_tokens=inp, output_tokens=out)


def read_anthropic_stream_usage(event: dict) -> tuple[int | None, int | None]:
    """Return (input, output) updates from one Anthropic SSE event. Missing side is None."""

    event_type = event.get("type")
    if event_type == "message_start":
        raw = (event.get("message") or {}).get("usage")
        if isinstance(raw, dict) and "input_tokens" in raw:
            return _as_int(raw.get("input_tokens")), None
        return None, None
    if event_type == "message_delta":
        raw = event.get("usage")
        if isinstance(raw, dict) and "output_tokens" in raw:
            return None, _as_int(raw.get("output_tokens"))
        return None, None
    if event_type == "message":
        raw = event.get("usage")
        if isinstance(raw, dict):
            inp = _as_int(raw.get("input_tokens"))
            out = _as_int(raw.get("output_tokens"))
            return inp if inp else None, out if out else None
    return None, None


def estimate_tokens(messages: list[ChatMessage], step: AgentStep) -> TokenUsage:
    incoming = context_chars(messages)
    outgoing = len(step.content or "")
    for call in step.calls:
        outgoing += len(str(call.get("toolName") or "")) + len(str(call.get("input") or ""))
    return TokenUsage(
        input_tokens=max(1, incoming // CHARS_PER_TOKEN) if incoming else 0,
        output_tokens=max(1, outgoing // CHARS_PER_TOKEN) if outgoing else 0,
    )


def price_for(model_name: str) -> tuple[float, float, str]:
    """Return (input_per_1m, output_per_1m, label)."""

    lowered = (model_name or "").strip().lower()
    if not lowered or lowered == "mock":
        return (0.0, 0.0, "mock")
    for needle, inp, out in _PRICES:
        if needle in lowered:
            return inp, out, needle
    return (*_DEFAULT_PRICE, "default")


def estimate_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    inp_rate, out_rate, _label = price_for(model_name)
    return (input_tokens / 1_000_000) * inp_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class UsageLedger:
    """Running tally for this chat. Survives /exit via the session file."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0
    last_model: str = ""
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        *,
        model: str,
        usage: TokenUsage | None,
        messages: list[ChatMessage],
        step: AgentStep,
    ) -> None:
        if usage is not None and (usage.input_tokens or usage.output_tokens):
            inp, out = usage.input_tokens, usage.output_tokens
            estimated = False
        else:
            guessed = estimate_tokens(messages, step)
            inp, out = guessed.input_tokens, guessed.output_tokens
            estimated = True
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        if estimated:
            self.estimated_calls += 1
        self.last_model = model
        bucket = self.by_model.setdefault(
            model or "unknown",
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_calls": 0},
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += inp
        bucket["output_tokens"] += out
        if estimated:
            bucket["estimated_calls"] += 1

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_calls": self.estimated_calls,
            "last_model": self.last_model,
            "by_model": dict(self.by_model),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> UsageLedger:
        raw = data if isinstance(data, dict) else {}
        by_model = raw.get("by_model") or {}
        cleaned: dict[str, dict[str, int]] = {}
        if isinstance(by_model, dict):
            for name, bucket in by_model.items():
                if isinstance(bucket, dict):
                    cleaned[str(name)] = {
                        "calls": _as_int(bucket.get("calls")),
                        "input_tokens": _as_int(bucket.get("input_tokens")),
                        "output_tokens": _as_int(bucket.get("output_tokens")),
                        "estimated_calls": _as_int(bucket.get("estimated_calls")),
                    }
        return cls(
            calls=_as_int(raw.get("calls")),
            input_tokens=_as_int(raw.get("input_tokens")),
            output_tokens=_as_int(raw.get("output_tokens")),
            estimated_calls=_as_int(raw.get("estimated_calls")),
            last_model=str(raw.get("last_model") or ""),
            by_model=cleaned,
        )

    def format(self, *, model_name: str = "", messages: list[ChatMessage] | None = None) -> str:
        used_model = model_name or self.last_model or "unknown"
        inp_rate, out_rate, price_label = price_for(used_model)
        cost = estimate_usd(used_model, self.input_tokens, self.output_tokens)
        lines = [
            f"model calls: {self.calls}",
            f"input tokens: {self.input_tokens}",
            f"output tokens: {self.output_tokens}",
        ]
        if messages is not None:
            lines.append(f"conversation: {context_chars(messages)} / {compact_limit()} characters")
        if used_model == "mock" or price_label == "mock":
            lines.append("estimated cost: $0.0000 (offline mock, not a real bill)")
        elif price_label == "default":
            lines.append(
                f"estimated cost: ${cost:.4f} (no listed rate for {used_model}; "
                f"used default ${inp_rate:.2f}/${out_rate:.2f} per 1M tokens)"
            )
        else:
            lines.append(
                f"estimated cost: ${cost:.4f} ({price_label} list: "
                f"${inp_rate:.2f}/${out_rate:.2f} per 1M input/output tokens)"
            )
        if self.estimated_calls:
            lines.append(
                f"note: {self.estimated_calls} call(s) had no API usage field; "
                f"counted as characters/{CHARS_PER_TOKEN}"
            )
        if len(self.by_model) > 1:
            lines.append("by model:")
            for name, bucket in self.by_model.items():
                lines.append(
                    f"  {name}: {bucket['calls']} call(s), "
                    f"{bucket['input_tokens']} in / {bucket['output_tokens']} out"
                )
        return "\n".join(lines)

    def status_line(self, model_name: str = "") -> str:
        used_model = model_name or self.last_model or "unknown"
        cost = estimate_usd(used_model, self.input_tokens, self.output_tokens)
        if used_model == "mock" or price_for(used_model)[2] == "mock":
            return f"usage: {self.calls} call(s), {self.input_tokens + self.output_tokens} tokens (mock, $0)"
        return f"usage: {self.calls} call(s), {self.input_tokens + self.output_tokens} tokens, ~${cost:.4f}"
