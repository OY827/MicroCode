import json
from pathlib import Path

from microcode.memory import ProjectMemory
from microcode.memory_reranker import (
    MemoryReranker,
    RerankResult,
    parse_rerank_json,
)
from microcode.types import AgentStep, ChatMessage


class _JsonModel:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []
        self.tools = object()
        self.stream = True

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        self.calls += 1
        self.seen.append(list(messages))
        return AgentStep(type="assistant", content=self.payload)


def test_parse_rerank_json_strips_fences() -> None:
    result = parse_rerank_json(
        '```json\n{"selected": ["a"], "rejected": [], "conflicts": [], "summary": "Use pytest."}\n```',
        {"a", "b"},
    )
    assert result is not None
    assert result.selected_ids == ["a"]
    assert result.summary == "Use pytest."


def test_parse_rerank_json_drops_unknown_ids() -> None:
    result = parse_rerank_json('{"selected": ["a", "nope"], "summary": "ok"}', {"a"})
    assert result is not None
    assert result.selected_ids == ["a"]


def test_parse_rerank_json_rejects_garbage() -> None:
    assert parse_rerank_json("not json", {"a"}) is None


def test_disabled_reranker_is_passthrough() -> None:
    reranker = MemoryReranker()
    assert reranker.enabled is False
    result = reranker.curate(
        [type("E", (), {"id": "a", "content": "Use pytest"})()],
        "run tests",
    )
    assert result.selected_ids == ["a"]
    assert result.confidence == 0.3


def test_curate_uses_model_selection() -> None:
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "selected": ["keep"],
                "rejected": [{"id": "drop", "reason": "unrelated"}],
                "conflicts": [],
                "summary": "Prefer pytest.",
            }
        )

    reranker = MemoryReranker(complete=complete)
    keep = type("E", (), {"id": "keep", "content": "Use pytest", "category": "testing", "tags": [], "usage_count": 0})()
    drop = type("E", (), {"id": "drop", "content": "DuckDuckGo", "category": "general", "tags": [], "usage_count": 0})()
    result = reranker.curate([keep, drop], "how should I run unit tests")
    assert result.selected_ids == ["keep"]
    assert result.summary == "Prefer pytest."
    assert calls and "how should I run unit tests" in calls[0]
    again = reranker.curate([keep, drop], "how should I run unit tests")
    assert again.selected_ids == ["keep"]
    assert len(calls) == 1
    assert reranker.cache_hits == 1


def test_curate_falls_back_when_complete_fails() -> None:
    reranker = MemoryReranker(complete=lambda _prompt: (_ for _ in ()).throw(RuntimeError("boom")))
    first = type("E", (), {"id": "a", "content": "one"})()
    second = type("E", (), {"id": "b", "content": "two"})()
    result = reranker.curate([first, second], "task")
    assert result.selected_ids == ["a", "b"]
    assert reranker.fallback_count == 1


def test_curate_empty_selected_takes_first_three() -> None:
    ids = [f"n{index}" for index in range(4)]
    payload = json.dumps({"selected": [], "summary": ""})
    reranker = MemoryReranker(complete=lambda _prompt: payload)
    candidates = [type("E", (), {"id": item, "content": item})() for item in ids]
    result = reranker.curate(candidates, "task")
    assert result.selected_ids == ["n0", "n1", "n2"]


def test_from_model_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_MEMORY_RERANK", "0")
    assert MemoryReranker.from_model(_JsonModel("{}")) is None


def test_from_model_skips_mock() -> None:
    from microcode.mock_model import MockModelAdapter

    assert MemoryReranker.from_model(MockModelAdapter()) is None
    from microcode.mock_model import MockModelAdapter

    assert MemoryReranker.from_model(MockModelAdapter()) is None


def test_complete_from_adapter_clears_tools_and_stream() -> None:
    from microcode.memory_reranker import complete_from_adapter

    model = _JsonModel('{"selected": ["a"]}')
    original_tools = model.tools
    text = complete_from_adapter(model, "pick one")
    assert text == '{"selected": ["a"]}'
    assert model.stream is True
    assert model.tools is original_tools
    assert model.seen[0][0]["role"] == "system"


def test_inject_rerank_keeps_selected_note(tmp_path: Path) -> None:
    notes = [
        "Prefer ruff for formatting",
        "Never commit .env files",
        "DuckDuckGo for web search",
        "AST finds Python symbols",
        "Git commit needs staged files",
        "Explore subagent is read only",
        "Checkpoint restores the last write",
        "Skills live under .microcode/skills",
        "MCP talks JSON-RPC over stdio",
        "Use pytest for unit tests",
    ]
    chosen: dict[str, str] = {}

    def complete(prompt: str) -> str:
        pytest_id = chosen["id"]
        return json.dumps(
            {
                "selected": [pytest_id],
                "rejected": [],
                "conflicts": [],
                "summary": "Run tests with pytest.",
            }
        )

    memory = ProjectMemory(tmp_path, reranker=MemoryReranker(complete=complete))
    for note in notes:
        entry = memory.manager.add(f"unit tests: {note}", category="general")
        if "pytest" in note:
            chosen["id"] = entry.id
    hits = memory.manager.search("how should I run unit tests", top_k=3, rerank=True)
    assert [entry.content for entry, _score in hits] == [f"unit tests: Use pytest for unit tests"]
    injected = memory.inject("You are MicroCode.", query="how should I run unit tests", top_k=3)
    assert "retrieved for this turn" in injected
    assert "pytest" in injected
    assert "Run tests with pytest." in injected
    assert "DuckDuckGo" not in injected
    assert "ruff" not in injected


def test_slash_search_does_not_call_reranker(tmp_path: Path) -> None:
    calls = {"n": 0}

    def complete(_prompt: str) -> str:
        calls["n"] += 1
        return json.dumps({"selected": [], "summary": ""})

    memory = ProjectMemory(tmp_path, reranker=MemoryReranker(complete=complete))
    for note in [
        "Prefer ruff for formatting",
        "Never commit .env files",
        "DuckDuckGo for web search",
        "AST finds Python symbols",
        "Git commit needs staged files",
        "Explore subagent is read only",
        "Checkpoint restores the last write",
        "Skills live under .microcode/skills",
        "MCP talks JSON-RPC over stdio",
        "Use pytest for unit tests",
    ]:
        memory.append(note)
    report = memory.format_search("how should I run unit tests")
    assert "pytest" in report
    assert calls["n"] == 0
    assert RerankResult.fallback(["a"]).selected_ids == ["a"]
