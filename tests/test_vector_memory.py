from microcode.vector_memory import (
    DenseVectorStore,
    SparseVectorStore,
    _load_minilm_encoder,
    classify_content,
    parse_notes,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_english_and_cjk() -> None:
    tokens = tokenize("Use pytest 跑测试")
    assert "pytest" in tokens
    assert "测试" in tokens


def test_parse_notes_splits_bullets() -> None:
    notes = parse_notes("# Project memory\n\n- Use pytest\n- Never commit .env\n")
    assert [note.content for note in notes] == ["Use pytest", "Never commit .env"]


def test_tfidf_ranks_related_note_first() -> None:
    notes = parse_notes(
        "\n".join(
            [
                "- Use pytest for unit tests",
                "- Prefer ruff to format Python",
                "- Never commit .env files",
                "- DuckDuckGo for web search",
                "- AST finds Python symbols",
                "- Git commit needs staged files",
                "- Explore subagent is read only",
                "- Checkpoint restores the last write",
                "- Skills live under .microcode/skills",
                "- MCP talks JSON-RPC over stdio",
            ]
        )
    )
    store = SparseVectorStore()
    store.index(notes)
    hits = store.search("how should I run unit tests", top_k=3)
    assert hits
    assert "pytest" in hits[0][0].content


def test_search_returns_empty_without_overlap() -> None:
    store = SparseVectorStore()
    store.index(parse_notes("- Never commit .env files\n"))
    assert store.search("zip hmac uuid") == []


def test_classify_testing_content() -> None:
    category, tags = classify_content("Use pytest for unit tests")
    assert category == "testing"
    assert "test" in tags


def test_reciprocal_rank_fusion_prefers_agreement() -> None:
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "d"])
    assert fused[0] in {"a", "c"}


def _topic_encoder(text: str) -> list[float]:
    lowered = text.lower()
    testing = 1.0 if any(word in lowered for word in ("pytest", "unit", "test", "harness", "checking")) else 0.0
    web = 1.0 if "duckduckgo" in lowered else 0.0
    return [testing, web, 0.05]


def test_dense_store_disabled_without_encoder() -> None:
    store = DenseVectorStore(load_default=False)
    assert store.enabled is False
    assert store.index(parse_notes("- Use pytest\n")) == 0
    assert store.search("unit tests") == []


def test_dense_load_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_DENSE_MEMORY", "0")
    assert _load_minilm_encoder() is None


def test_dense_ranks_semantic_match_without_shared_words() -> None:
    notes = parse_notes(
        "\n".join(
            [
                "- Always execute the checking harness before merging",
                "- DuckDuckGo for public documentation",
                "- Never commit .env files",
            ]
        )
    )
    store = DenseVectorStore(encoder=_topic_encoder)
    assert store.enabled is True
    assert store.index(notes) == 3
    hits = store.search("how should I run unit tests", top_k=2)
    assert hits
    assert "checking harness" in hits[0][0].content


def test_dense_skips_notes_when_encoder_fails() -> None:
    def boom(text: str) -> list[float]:
        if "boom" in text:
            raise RuntimeError("encode failed")
        return [1.0, 0.0]

    store = DenseVectorStore(encoder=boom)
    notes = parse_notes("- keep this\n- boom note\n")
    assert store.index(notes) == 1
    hits = store.search("keep this")
    assert len(hits) == 1
    assert hits[0][0].content == "keep this"


def test_dense_forget_except_drops_stale_embeddings() -> None:
    store = DenseVectorStore(encoder=_topic_encoder)
    notes = parse_notes("- Always execute the checking harness\n- DuckDuckGo for docs\n")
    store.index(notes)
    store.forget_except({notes[0].id})
    hits = store.search("DuckDuckGo documentation")
    assert hits == []


