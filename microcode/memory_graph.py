from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from microcode.vector_memory import tokenize

GRAPH_VERSION = 1
GRAPH_ALPHA = 0.85
GRAPH_PPR_ITERS = 12
GRAPH_MAX_HOPS = 2
GRAPH_SEED_LIMIT = 16
SIMILAR_THRESHOLD = 0.35
SAME_AS_THRESHOLD = 0.9
SUPERSEDE_THRESHOLD = 0.15

RELATION_WEIGHT = {
    "supersedes": 0.90,
    "same_as": 0.75,
    "similar": 0.35,
}
REPLACEMENT_MARKERS = (
    "replace",
    "replaced",
    "supersede",
    "migrat",
    "instead",
    "switch",
    "改用",
    "替换",
    "迁移",
    "改为",
    "取代",
)
GRAPH_QUERY_MARKERS = (
    "why",
    "because",
    "instead",
    "replace",
    "replaced",
    "supersede",
    "migrat",
    "related",
    "contradict",
    "depend",
    "previous",
    "what led",
    "what changed",
    "which decision",
    "为什么",
    "原因",
    "改用",
    "替换",
    "相关",
    "之前",
)


def graph_disabled() -> bool:
    raw = (os.environ.get("MICROCODE_MEMORY_GRAPH") or "").strip().lower()
    return raw in {"0", "false", "no", "off"}


def graph_shaped_query(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in GRAPH_QUERY_MARKERS)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _has_replacement(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REPLACEMENT_MARKERS)


@dataclass
class MemoryFact:
    id: str
    memory_id: str
    subject: str
    predicate: str
    value: str
    confidence: float = 0.5

    def searchable_text(self) -> str:
        return " ".join([self.subject, self.predicate, self.value])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryFact:
        return cls(
            id=str(data.get("id") or ""),
            memory_id=str(data.get("memory_id") or ""),
            subject=str(data.get("subject") or ""),
            predicate=str(data.get("predicate") or "memory"),
            value=str(data.get("value") or ""),
            confidence=float(data.get("confidence") or 0.5),
        )


@dataclass
class MemoryEdge:
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEdge:
        return cls(
            id=str(data.get("id") or ""),
            source_id=str(data.get("source_id") or ""),
            target_id=str(data.get("target_id") or ""),
            relation=str(data.get("relation") or "similar"),
            weight=float(data.get("weight") or 0.35),
        )


@dataclass(frozen=True)
class GraphHit:
    memory_id: str
    score: float
    relations: tuple[str, ...] = ()


class MemoryGraphStore:
    """Sidecar fact graph: lexical seeds, then a short Personalized PageRank walk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.facts: dict[str, MemoryFact] = {}
        self.edges: dict[str, MemoryEdge] = {}
        self._fingerprint: tuple[tuple[str, str], ...] = ()
        if self.path is not None:
            self._load()

    @property
    def enabled(self) -> bool:
        return bool(self.facts) and not graph_disabled()

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def clear(self) -> None:
        self.facts.clear()
        self.edges.clear()
        self._fingerprint = ()

    def sync_entries(self, entries: list) -> bool:
        """Turn memory notes into facts and rebuild typed links."""

        live = [entry for entry in entries if getattr(entry, "id", "") and getattr(entry, "content", "")]
        if not live:
            changed = bool(self.facts or self.edges)
            self.clear()
            if changed and self.path is not None and self.path.is_file():
                self.save()
            return changed
        fingerprint = tuple(
            (str(entry.id), str(entry.content), str(getattr(entry, "category", "")), tuple(getattr(entry, "tags", ()) or ()))
            for entry in sorted(live, key=lambda item: str(item.id))
        )
        if fingerprint == self._fingerprint and self.facts:
            return False
        self.facts.clear()
        self.edges.clear()
        for entry in live:
            fact_id = f"memory:{entry.id}"
            self.facts[fact_id] = MemoryFact(
                id=fact_id,
                memory_id=str(entry.id),
                subject=str(getattr(entry, "category", "") or "memory"),
                predicate="memory",
                value=str(entry.content),
                confidence=0.5,
            )
        self._link(live)
        self._fingerprint = fingerprint
        self.save()
        return True

    def search(self, query: str, *, limit: int = 10, max_hops: int = GRAPH_MAX_HOPS) -> list[GraphHit]:
        if not self.enabled or not query.strip() or limit <= 0:
            return []
        query_tokens = tokenize(query, drop_stopwords=True)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        lexical: dict[str, float] = {}
        for fact_id, fact in self.facts.items():
            score = _jaccard(query_set, set(tokenize(fact.searchable_text(), drop_stopwords=True)))
            if score > 0.0:
                lexical[fact_id] = score
        if not lexical:
            return []
        ranked_seeds = sorted(lexical.items(), key=lambda item: item[1], reverse=True)
        floor = max(0.08, ranked_seeds[0][1] * 0.25)
        seeds = {
            fact_id
            for fact_id, _score in ranked_seeds[:GRAPH_SEED_LIMIT]
            if _score >= floor
        }
        if not seeds:
            return []
        adjacency = self._adjacency()
        reachable = _reachable(seeds, adjacency, max_hops=max_hops)
        teleport_total = sum(lexical[fact_id] for fact_id in seeds) or 1.0
        teleport = {fact_id: lexical[fact_id] / teleport_total for fact_id in seeds}
        rank = dict(teleport)
        for _ in range(GRAPH_PPR_ITERS):
            nxt = {fact_id: (1.0 - GRAPH_ALPHA) * teleport.get(fact_id, 0.0) for fact_id in reachable}
            for fact_id, mass in rank.items():
                neighbours = [(other, edge) for other, edge in adjacency.get(fact_id, ()) if other in reachable]
                if not neighbours:
                    nxt[fact_id] = nxt.get(fact_id, 0.0) + GRAPH_ALPHA * mass
                    continue
                total = sum(RELATION_WEIGHT.get(edge.relation, edge.weight) for _other, edge in neighbours) or 1.0
                for other, edge in neighbours:
                    weight = RELATION_WEIGHT.get(edge.relation, edge.weight)
                    nxt[other] = nxt.get(other, 0.0) + GRAPH_ALPHA * mass * weight / total
            rank = nxt
        peak = max(rank.values(), default=1.0) or 1.0
        by_memory: dict[str, float] = {}
        relations_by_memory: dict[str, set[str]] = defaultdict(set)
        for fact_id, fact in self.facts.items():
            if fact_id not in reachable or not fact.memory_id:
                continue
            score = 0.55 * (rank.get(fact_id, 0.0) / peak) + 0.45 * lexical.get(fact_id, 0.0)
            if score > by_memory.get(fact.memory_id, 0.0):
                by_memory[fact.memory_id] = score
            for _other, edge in adjacency.get(fact_id, ()):
                relations_by_memory[fact.memory_id].add(edge.relation)
        hits = [
            GraphHit(memory_id=memory_id, score=score, relations=tuple(sorted(relations_by_memory.get(memory_id, ()))))
            for memory_id, score in by_memory.items()
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GRAPH_VERSION,
            "updated_at": time.time(),
            "facts": [fact.to_dict() for fact in self.facts.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("version") != GRAPH_VERSION:
            return
        for item in payload.get("facts") or []:
            if isinstance(item, dict) and item.get("id") and item.get("memory_id"):
                fact = MemoryFact.from_dict(item)
                self.facts[fact.id] = fact
        for item in payload.get("edges") or []:
            if not isinstance(item, dict):
                continue
            edge = MemoryEdge.from_dict(item)
            if edge.source_id in self.facts and edge.target_id in self.facts:
                self.edges[edge.id] = edge

    def _link(self, entries: list) -> None:
        tokens = {
            entry.id: set(tokenize(f"{entry.content} {getattr(entry, 'category', '')} {' '.join(getattr(entry, 'tags', []) or [])}", drop_stopwords=True))
            for entry in entries
        }
        for index, left in enumerate(entries):
            for right in entries[index + 1 :]:
                score = _jaccard(tokens[left.id], tokens[right.id])
                left_fact = f"memory:{left.id}"
                right_fact = f"memory:{right.id}"
                if _has_replacement(left.content) and not _has_replacement(right.content) and score >= SUPERSEDE_THRESHOLD:
                    self._add_edge(left_fact, right_fact, "supersedes", RELATION_WEIGHT["supersedes"])
                    continue
                if _has_replacement(right.content) and not _has_replacement(left.content) and score >= SUPERSEDE_THRESHOLD:
                    self._add_edge(right_fact, left_fact, "supersedes", RELATION_WEIGHT["supersedes"])
                    continue
                if score >= SAME_AS_THRESHOLD:
                    self._add_edge(left_fact, right_fact, "same_as", RELATION_WEIGHT["same_as"], both=True)
                    continue
                if score >= SIMILAR_THRESHOLD:
                    self._add_edge(left_fact, right_fact, "similar", RELATION_WEIGHT["similar"], both=True)
                    continue
                left_cat = str(getattr(left, "category", "") or "general")
                right_cat = str(getattr(right, "category", "") or "general")
                if left_cat != "general" and left_cat == right_cat:
                    self._add_edge(left_fact, right_fact, "similar", 0.25, both=True)

    def _add_edge(self, source_id: str, target_id: str, relation: str, weight: float, *, both: bool = False) -> None:
        edge_id = f"{relation}:{source_id}->{target_id}"
        self.edges[edge_id] = MemoryEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )
        if both:
            back_id = f"{relation}:{target_id}->{source_id}"
            self.edges[back_id] = MemoryEdge(
                id=back_id,
                source_id=target_id,
                target_id=source_id,
                relation=relation,
                weight=weight,
            )

    def _adjacency(self) -> dict[str, list[tuple[str, MemoryEdge]]]:
        graph: dict[str, list[tuple[str, MemoryEdge]]] = defaultdict(list)
        for edge in self.edges.values():
            graph[edge.source_id].append((edge.target_id, edge))
        return graph


def _reachable(
    seeds: set[str],
    adjacency: dict[str, list[tuple[str, MemoryEdge]]],
    *,
    max_hops: int,
) -> set[str]:
    seen = set(seeds)
    queue: deque[tuple[str, int]] = deque((node, 0) for node in seeds)
    while queue:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for neighbour, _edge in adjacency.get(node, ()):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append((neighbour, hops + 1))
    return seen
