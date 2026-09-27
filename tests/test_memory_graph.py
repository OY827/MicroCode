from pathlib import Path

from microcode.memory import MemoryScope, ProjectMemory
from microcode.memory_graph import (
    MemoryGraphStore,
    graph_shaped_query,
)


def test_graph_shaped_query_detects_relational_wording() -> None:
    assert graph_shaped_query("why did we replace Redux")
    assert graph_shaped_query("改用 Zustand 的原因")
    assert not graph_shaped_query("how should I run unit tests")


def test_sync_links_replacement_as_supersedes(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, graph=MemoryGraphStore())
    old = memory.manager.add("Use Redux for client state", category="architecture")
    new = memory.manager.add("Migrated from Redux to Zustand for client state", category="architecture")
    graph = memory.manager._graph
    assert graph.edge_count > 0
    relations = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges.values()}
    assert (f"memory:{new.id}", f"memory:{old.id}", "supersedes") in relations


def test_graph_search_reaches_linked_decision(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.manager.add("Use Redux for client state", category="architecture")
    memory.manager.add("Migrated from Redux to Zustand for client state", category="architecture")
    hits = memory.manager._graph.search("why did we replace Redux")
    contents = [
        entry.content
        for entry in memory.manager.stores[MemoryScope.PROJECT].entries
        if any(hit.memory_id == entry.id for hit in hits)
    ]
    assert any("Zustand" in text for text in contents)
    assert any("Redux" in text for text in contents)
    assert (tmp_path / ".microcode" / "memory_graph.json").is_file()


def test_search_fuses_graph_for_why_query(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.manager.add("Use Redux for client state", category="architecture")
    memory.manager.add("Migrated from Redux to Zustand for client state", category="architecture")
    memory.manager.add("DuckDuckGo for public documentation", category="general")
    hits = memory.manager.search("why did we replace Redux")
    texts = [entry.content for entry, _score in hits]
    assert any("Zustand" in text for text in texts)
    assert all("DuckDuckGo" not in text for text in texts[:2])


def test_graph_persists_across_reload(tmp_path: Path) -> None:
    first = ProjectMemory(tmp_path)
    first.manager.add("Use Redux for client state", category="architecture")
    first.manager.add("Migrated from Redux to Zustand for client state", category="architecture")
    edge_count = first.manager._graph.edge_count
    reloaded = ProjectMemory(tmp_path)
    assert reloaded.manager._graph.edge_count == edge_count
    assert reloaded.manager._graph.facts


def test_graph_disabled_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_MEMORY_GRAPH", "0")
    memory = ProjectMemory(tmp_path)
    memory.manager.add("Use Redux for client state", category="architecture")
    memory.manager.add("Migrated from Redux to Zustand for client state", category="architecture")
    assert memory.manager._graph.search("why did we replace Redux") == []
