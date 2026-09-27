import json
from pathlib import Path

import pytest

from microcode.memory import (
    ARCHIVE_SECONDS,
    MAX_SCOPE_ENTRIES,
    PROMOTE_AGE_SECONDS,
    WORKING_SECONDS,
    MemoryEntry,
    MemoryScope,
    MemoryTier,
    ProjectMemory,
    apply_system_message,
    parse_memory_command,
)
from microcode.vector_memory import DenseVectorStore
from microcode.repl import run_repl, run_user_turn
from microcode.session import SessionStore
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
                run=lambda input_data, _context: ToolResult(ok=True, output="ok"),
            )
        ]
    )


@pytest.fixture(autouse=True)
def _isolate_memory_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MICROCODE_HOME", str(tmp_path / "home"))


def test_read_missing_memory_is_empty(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    assert memory.read() == ""
    assert "No project memory yet" in memory.display()


def test_inject_appends_memory_section(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    injected = memory.inject("You are MicroCode.")
    assert injected.startswith("You are MicroCode.")
    assert "## Project memory" in injected
    assert "Use pytest" in injected


def test_inject_empty_leaves_system_prompt(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    assert memory.inject("You are MicroCode.") == "You are MicroCode."


def test_inject_truncates_long_memory(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, max_chars=20)
    memory.append("x" * 100)
    injected = memory.inject("sys")
    assert "memory truncated" in injected
    assert len(memory.read()) > 20


def test_append_creates_markdown_file(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    report = memory.append("Prefer read_file over guessing")
    assert report.startswith("Remembered")
    text = (tmp_path / ".microcode" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- Prefer read_file over guessing" in text


def test_apply_system_message_replaces_leading_system() -> None:
    messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    updated = apply_system_message(messages, "new")
    assert updated[0] == {"role": "system", "content": "new"}
    assert updated[1]["content"] == "hi"


def test_parse_memory_command() -> None:
    assert parse_memory_command("/memory") == ("show", "")
    assert parse_memory_command("/memory add Use pytest") == ("add", "Use pytest")
    assert parse_memory_command("/memory add user: Prefer ruff") == ("add", "user: Prefer ruff")
    assert parse_memory_command("/memory add") == ("add", "")
    assert parse_memory_command("/memory search pytest") == ("search", "pytest")
    assert parse_memory_command("/memory search") == ("search", "")
    assert parse_memory_command("/memory maintain") == ("maintain", "")
    assert parse_memory_command("/memory weird") == ("help", "")


def test_search_formats_ranked_notes(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest for unit tests")
    memory.append("Never commit .env")
    report = memory.format_search("run unit tests")
    assert "pytest" in report
    assert report.splitlines()[0].split("  ", 1)[1]


def test_inject_retrieves_when_memory_is_large(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    filler = [
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
    for note in filler:
        memory.append(note)
    injected = memory.inject("You are MicroCode.", query="how should I run unit tests", top_k=3)
    assert "retrieved for this turn" in injected
    assert "pytest" in injected
    assert "DuckDuckGo" not in injected


def test_user_turn_retrieves_memory_into_system(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
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
    model = ScriptedModel([AgentStep(type="assistant", content="ok")])
    run_user_turn(
        model=model,
        tools=_echo_registry(),
        messages=[{"role": "system", "content": "You are MicroCode."}],
        user_text="how should I run unit tests",
        cwd=str(tmp_path),
        system_prompt="You are MicroCode.",
    )
    system = str(model.seen[0][0]["content"])
    assert "retrieved for this turn" in system
    assert "pytest" in system



def test_repl_memory_add_refreshes_system(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    memory = ProjectMemory(tmp_path)
    base = "You are MicroCode."
    session = store.create(str(tmp_path), [{"role": "system", "content": base}])
    store.save(session)
    lines = iter(["/memory add Use pytest", "hello", "/exit"])
    model = ScriptedModel([AgentStep(type="assistant", content="ok")])

    run_repl(
        model=model,
        tools=_echo_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        memory=memory,
        system_prompt=base,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    loaded = store.load(session.id)
    assert "Use pytest" in str(loaded.messages[0]["content"])
    assert loaded.messages[1]["content"] == "hello"
    assert "Use pytest" in str(model.seen[0][0]["content"])
    assert (tmp_path / ".microcode" / "MEMORY.md").is_file()


def test_user_and_local_scopes_stay_separate(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, home=tmp_path / "home")
    memory.append("user: Prefer dark themes")
    memory.append("local: Machine has no GPU")
    memory.append("Shared project convention")
    listing = memory.display()
    assert "Prefer dark themes" in listing
    assert "Machine has no GPU" in listing
    assert (tmp_path / ".microcode" / "memory.json").is_file()
    assert (tmp_path / "home" / "memory" / "memory.json").is_file()
    assert (tmp_path / ".microcode" / "memory-local" / "memory.json").is_file()
    reloaded = ProjectMemory(tmp_path, home=tmp_path / "home")
    assert any(entry.content == "Prefer dark themes" for entry in reloaded.manager.stores[MemoryScope.USER].entries)
    assert any(entry.content == "Machine has no GPU" for entry in reloaded.manager.stores[MemoryScope.LOCAL].entries)


def test_legacy_markdown_imports_into_project_scope(tmp_path: Path) -> None:
    md = tmp_path / ".microcode" / "MEMORY.md"
    md.parent.mkdir(parents=True)
    md.write_text("# Project memory\n\n- Use pytest\n", encoding="utf-8")
    memory = ProjectMemory(tmp_path, home=tmp_path / "home")
    imported = [entry for entry in memory.manager.stores[MemoryScope.PROJECT].entries if entry.content == "Use pytest"]
    assert imported
    assert imported[0].tier == MemoryTier.SHORT_TERM


def test_new_note_starts_as_working(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    report = memory.append("Use pytest")
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    assert entry.tier == MemoryTier.WORKING
    assert report.startswith("Remembered (project/working):")
    listing = memory.manager.format_list()
    assert "[working]" in listing


def test_legacy_json_without_tier_is_short_term(tmp_path: Path) -> None:
    path = tmp_path / ".microcode" / "memory.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "scope": "project",
                "entries": [
                    {
                        "id": "legacy-1",
                        "content": "Old convention",
                        "created_at": 1.0,
                        "updated_at": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    memory = ProjectMemory(tmp_path)
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    assert entry.content == "Old convention"
    assert entry.tier == MemoryTier.SHORT_TERM


def test_session_end_demotes_working_to_short_term(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    report = memory.maintain(session_end=True)
    assert entry.tier == MemoryTier.SHORT_TERM
    assert "short_term=1" in report
    reloaded = ProjectMemory(tmp_path)
    assert reloaded.manager.stores[MemoryScope.PROJECT].entries[0].tier == MemoryTier.SHORT_TERM


def test_idle_short_term_archives_but_keeps_content(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    long_note = "Prefer read_file over guessing. " * 8
    memory.append(long_note.strip())
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    entry.tier = MemoryTier.SHORT_TERM
    entry.usage_count = 0
    created = 1_000_000.0
    entry.created_at = created
    entry.last_accessed = created
    original = entry.content
    report = memory.maintain(now=created + ARCHIVE_SECONDS + 1)
    assert "archival=1" in report
    assert entry.tier == MemoryTier.ARCHIVAL
    assert entry.content == original
    assert entry.summary
    assert entry.summary != original
    injected = memory.inject("You are MicroCode.")
    assert entry.summary in injected
    assert original not in injected
    saved = json.loads((tmp_path / ".microcode" / "memory.json").read_text(encoding="utf-8"))
    assert saved["entries"][0]["content"] == original
    assert saved["entries"][0]["summary"] == entry.summary


def test_used_notes_promote_to_long_term(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    entry.tier = MemoryTier.SHORT_TERM
    entry.usage_count = 3
    entry.created_at = 1.0
    entry.last_accessed = 1.0
    report = memory.maintain(now=1.0 + PROMOTE_AGE_SECONDS + 1)
    assert "long_term=1" in report
    assert entry.tier == MemoryTier.LONG_TERM


def test_idle_long_term_archives(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Stable project convention")
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    entry.tier = MemoryTier.LONG_TERM
    entry.usage_count = 5
    entry.last_accessed = 1.0
    memory.maintain(now=1.0 + ARCHIVE_SECONDS + 1)
    assert entry.tier == MemoryTier.ARCHIVAL
    assert entry.content == "Stable project convention"


def test_search_bumps_usage_and_wakes_archival(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest for unit tests")
    memory.append("Never commit .env files")
    pytest_note = next(entry for entry in memory.manager.all_entries() if "pytest" in entry.content)
    pytest_note.tier = MemoryTier.ARCHIVAL
    pytest_note.summary = "Use pytest"
    pytest_note.usage_count = 0
    report = memory.format_search("pytest")
    assert pytest_note.usage_count == 1
    assert pytest_note.tier == MemoryTier.SHORT_TERM
    assert "[project/short_term]" in report
    assert "pytest" in report


def test_evicts_archival_before_hot_notes(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    store = memory.manager.stores[MemoryScope.PROJECT]
    store.add(
        MemoryEntry(
            id="old-arch",
            scope=MemoryScope.PROJECT,
            category="general",
            content="cold archival note",
            tier=MemoryTier.ARCHIVAL,
            last_accessed=1.0,
        )
    )
    for index in range(MAX_SCOPE_ENTRIES):
        memory.manager.add(f"hot note {index}")
    ids = [entry.id for entry in store.entries]
    assert "old-arch" not in ids
    assert len(ids) == MAX_SCOPE_ENTRIES


def test_working_ages_into_short_term(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    entry.created_at = 1.0
    memory.maintain(now=1.0 + WORKING_SECONDS + 1)
    assert entry.tier == MemoryTier.SHORT_TERM


def test_exit_demotes_working_notes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    base = "You are MicroCode."
    session = store.create(str(tmp_path), [{"role": "system", "content": base}])
    store.save(session)
    lines = iter(["/memory maintain", "/exit"])

    run_repl(
        model=ScriptedModel([]),
        tools=_echo_registry(),
        messages=session.messages,
        cwd=str(tmp_path),
        session=session,
        store=store,
        memory=memory,
        system_prompt=base,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )

    entry = memory.manager.stores[MemoryScope.PROJECT].entries[0]
    assert entry.tier == MemoryTier.SHORT_TERM


def _topic_encoder(text: str) -> list[float]:
    lowered = text.lower()
    testing = 1.0 if any(word in lowered for word in ("pytest", "unit", "test", "harness", "checking")) else 0.0
    web = 1.0 if "duckduckgo" in lowered else 0.0
    return [testing, web, 0.05]


def test_search_uses_dense_when_words_do_not_overlap(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, dense=DenseVectorStore(encoder=_topic_encoder))
    memory.manager.add("Always execute the checking harness before merging", category="general")
    memory.manager.add("DuckDuckGo for public documentation", category="general")
    lexical = ProjectMemory(tmp_path / "lexical")
    lexical.manager.add("Always execute the checking harness before merging", category="general")
    lexical.manager.add("DuckDuckGo for public documentation", category="general")
    assert lexical.manager.search("how should I run unit tests") == []
    hits = memory.manager.search("how should I run unit tests")
    assert hits
    assert "checking harness" in hits[0][0].content


