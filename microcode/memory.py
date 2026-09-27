from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from microcode.types import ChatMessage
from microcode.vector_memory import (
    DenseVectorStore,
    MemoryNote,
    SparseVectorStore,
    bm25_score,
    classify_content,
    compute_avgdl,
    compute_idf,
    parse_notes,
    reciprocal_rank_fusion,
    tokenize,
)
from microcode.memory_graph import MemoryGraphStore, graph_shaped_query
from microcode.memory_reranker import RERANK_CANDIDATES, MemoryReranker

MEMORY_NAME = "MEMORY.md"
MEMORY_JSON = "memory.json"
DEFAULT_MAX_CHARS = 8_000
DEFAULT_TOP_K = 8
MAX_SCOPE_ENTRIES = 200
WORKING_SECONDS = 4 * 3600
PROMOTE_USES = 3
PROMOTE_AGE_SECONDS = 7 * 86400
ARCHIVE_SECONDS = 30 * 86400
SUMMARY_MAX_CHARS = 120
MEMORY_HEADING = "## Project memory"
SCOPE_PREFIX = re.compile(r"^(user|project|local)\s*:\s*(.+)$", re.IGNORECASE)


class MemoryScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


class MemoryTier(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVAL = "archival"


TIER_WEIGHT = {
    MemoryTier.WORKING: 1.0,
    MemoryTier.SHORT_TERM: 0.9,
    MemoryTier.LONG_TERM: 0.7,
    MemoryTier.ARCHIVAL: 0.4,
}
TIER_EVICT = {
    MemoryTier.ARCHIVAL: 0,
    MemoryTier.LONG_TERM: 1,
    MemoryTier.SHORT_TERM: 2,
    MemoryTier.WORKING: 3,
}


def summarize_text(content: str, max_len: int = SUMMARY_MAX_CHARS) -> str:
    cleaned = content.strip()
    if len(cleaned) <= max_len:
        return cleaned
    for separator in (". ", "。", "\n"):
        index = cleaned.find(separator)
        if 20 <= index <= max_len:
            return cleaned[: index + (0 if separator == "。" else 1)].strip()
    return cleaned[:max_len].rstrip() + "..."


@dataclass
class MemoryEntry:
    id: str
    scope: MemoryScope
    category: str
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0
    tier: MemoryTier = MemoryTier.WORKING
    summary: str = ""

    def prompt_text(self) -> str:
        if self.tier == MemoryTier.ARCHIVAL and self.summary:
            return self.summary
        return self.content

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "category": self.category,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "tags": list(self.tags),
            "usage_count": self.usage_count,
            "tier": self.tier.value,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        scope_raw = data.get("scope", "project")
        try:
            scope = MemoryScope(scope_raw)
        except ValueError:
            scope = MemoryScope.PROJECT
        created = float(data.get("created_at") or time.time())
        updated = float(data.get("updated_at") or created)
        try:
            tier = MemoryTier(str(data.get("tier") or MemoryTier.SHORT_TERM.value))
        except ValueError:
            tier = MemoryTier.SHORT_TERM
        return cls(
            id=str(data.get("id") or ""),
            scope=scope,
            category=str(data.get("category") or "general"),
            content=str(data.get("content") or ""),
            created_at=created,
            updated_at=updated,
            last_accessed=float(data.get("last_accessed") or updated),
            tags=[str(tag) for tag in data.get("tags") or [] if isinstance(tag, str)],
            usage_count=int(data.get("usage_count") or 0),
            tier=tier,
            summary=str(data.get("summary") or ""),
        )

    def as_note(self) -> MemoryNote:
        extra = " ".join([self.category, *self.tags])
        return MemoryNote(id=self.id, content=f"{self.content} {extra}".strip())


@dataclass
class MemoryStore:
    scope: MemoryScope
    entries: list[MemoryEntry] = field(default_factory=list)

    def format_markdown(self) -> str:
        if not self.entries:
            return f"# {self.scope.value.title()} memory\n"
        lines = [f"# {self.scope.value.title()} memory", ""]
        grouped: dict[str, list[MemoryEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.category, []).append(entry)
        for category, items in grouped.items():
            lines.append(f"## {category}")
            for entry in items:
                lines.append(f"- {entry.prompt_text()}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def format_prompt(self, *, sort_by_tier: bool = False) -> str:
        """Markdown for the system prompt: archival notes emit their summary."""

        entries = self.entries
        if sort_by_tier:
            entries = sorted(
                entries,
                key=lambda item: (-TIER_WEIGHT.get(item.tier, 0), -item.last_accessed),
            )
        if not entries:
            return f"# {self.scope.value.title()} memory\n"
        lines = [f"# {self.scope.value.title()} memory", ""]
        for entry in entries:
            lines.append(f"- {entry.prompt_text()}")
        return "\n".join(lines).rstrip() + "\n"

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        while len(self.entries) > MAX_SCOPE_ENTRIES:
            victim = min(
                self.entries,
                key=lambda item: (TIER_EVICT[item.tier], item.last_accessed),
            )
            self.entries.remove(victim)


class MemoryManager:
    """Three scopes (user/project/local) and four heat tiers on disk."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        home: str | Path | None = None,
        dense: DenseVectorStore | None = None,
        reranker: MemoryReranker | None = None,
        graph: MemoryGraphStore | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.home = Path(home) if home is not None else _default_home()
        self.stores: dict[MemoryScope, MemoryStore] = {
            scope: MemoryStore(scope=scope) for scope in MemoryScope
        }
        self._dense = dense if dense is not None else DenseVectorStore()
        self._reranker = reranker
        self._last_rerank_summary = ""
        self._load_all()
        self._graph = graph if graph is not None else MemoryGraphStore(self.workspace / ".microcode" / "memory_graph.json")
        self._graph.sync_entries(self.all_entries())

    def scope_dir(self, scope: MemoryScope) -> Path:
        if scope == MemoryScope.USER:
            return self.home / "memory"
        if scope == MemoryScope.LOCAL:
            return self.workspace / ".microcode" / "memory-local"
        return self.workspace / ".microcode"

    def markdown_path(self, scope: MemoryScope) -> Path:
        if scope == MemoryScope.PROJECT:
            return self.workspace / ".microcode" / MEMORY_NAME
        return self.scope_dir(scope) / MEMORY_NAME

    def json_path(self, scope: MemoryScope) -> Path:
        if scope == MemoryScope.PROJECT:
            return self.workspace / ".microcode" / MEMORY_JSON
        return self.scope_dir(scope) / MEMORY_JSON

    def _load_all(self) -> None:
        for scope in MemoryScope:
            self._load_scope(scope)

    def _load_scope(self, scope: MemoryScope) -> None:
        store = MemoryStore(scope=scope)
        json_file = self.json_path(scope)
        md_file = self.markdown_path(scope)
        if json_file.is_file():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            raw_entries = data.get("entries") if isinstance(data, dict) else None
            if isinstance(raw_entries, list):
                for item in raw_entries:
                    if not isinstance(item, dict) or not item.get("content"):
                        continue
                    entry = MemoryEntry.from_dict(item)
                    if not entry.id:
                        continue
                    entry.scope = scope
                    store.add(entry)
        elif md_file.is_file():
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                text = ""
            for index, note in enumerate(parse_notes(text), start=1):
                store.add(
                    MemoryEntry(
                        id=f"{scope.value}-md-{index}",
                        scope=scope,
                        category="general",
                        content=note.content,
                        tier=MemoryTier.SHORT_TERM,
                    )
                )
        self.stores[scope] = store

    def _save_scope(self, scope: MemoryScope) -> None:
        directory = self.scope_dir(scope)
        directory.mkdir(parents=True, exist_ok=True)
        store = self.stores[scope]
        payload = {
            "scope": scope.value,
            "last_updated": time.time(),
            "entries": [entry.to_dict() for entry in store.entries],
        }
        _atomic_write(self.json_path(scope), json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        _atomic_write(self.markdown_path(scope), store.format_markdown())

    def all_entries(self) -> list[MemoryEntry]:
        return [entry for scope in MemoryScope for entry in self.stores[scope].entries]

    def add(
        self,
        content: str,
        *,
        scope: MemoryScope = MemoryScope.PROJECT,
        category: str = "auto",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("content is required")
        final_category = category
        final_tags = list(tags or [])
        if category == "auto":
            final_category, auto_tags = classify_content(cleaned)
            final_tags = list(dict.fromkeys(final_tags + auto_tags))
        entry = MemoryEntry(
            id=f"{scope.value}-{uuid.uuid4().hex[:12]}",
            scope=scope,
            category=final_category or "general",
            content=cleaned,
            tags=final_tags,
        )
        self.stores[scope].add(entry)
        self._save_scope(scope)
        self._graph.sync_entries(self.all_entries())
        return entry

    def upsert_tagged(
        self,
        content: str,
        *,
        tag: str,
        extra_tags: list[str] | None = None,
        scope: MemoryScope = MemoryScope.PROJECT,
    ) -> tuple[MemoryEntry, bool]:
        """Replace the note that already has this tag, or add a new one."""

        cleaned = content.strip()
        if not cleaned:
            raise ValueError("content is required")
        for entry in self.all_entries():
            if tag in entry.tags:
                entry.content = cleaned
                entry.updated_at = time.time()
                entry.last_accessed = time.time()
                entry.tier = MemoryTier.WORKING
                entry.summary = ""
                self._save_scope(entry.scope)
                self._graph.sync_entries(self.all_entries())
                return entry, False
        tags = list(dict.fromkeys([tag, *(extra_tags or [])]))
        return self.add(cleaned, scope=scope, tags=tags), True

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        pool = list(self.stores[scope].entries) if scope else self.all_entries()
        if not pool or not query.strip():
            return []
        by_id = {entry.id: entry for entry in pool}
        notes = [entry.as_note() for entry in pool]
        bm25_ids = _bm25_rank(pool, query)
        store = SparseVectorStore()
        store.index(notes)
        vector_ids = [note.id for note, _score in store.search(query, top_k=max(top_k, len(pool)))]
        dense_ids: list[str] = []
        if self._dense.enabled:
            self._dense.index(notes)
            self._dense.forget_except({entry.id for entry in self.all_entries()})
            dense_ids = [
                note.id
                for note, _score in self._dense.search(
                    query,
                    top_k=max(top_k, len(pool)),
                    candidate_ids={note.id for note in notes},
                )
            ]
        graph_ids: list[str] = []
        if self._should_use_graph(query, bm25_ids, vector_ids, dense_ids):
            graph_ids = [hit.memory_id for hit in self._graph.search(query, limit=max(top_k, 10))]
        fused = reciprocal_rank_fusion(bm25_ids, vector_ids, dense_ids, graph_ids)
        weighted: list[tuple[float, MemoryEntry]] = []
        for index, entry_id in enumerate(fused):
            entry = by_id.get(entry_id)
            if entry is None:
                continue
            score = (1.0 / (index + 1)) * TIER_WEIGHT.get(entry.tier, 0.5)
            weighted.append((score, entry))
        weighted.sort(key=lambda item: item[0], reverse=True)
        reranker = self._reranker
        should_rerank = bool(rerank and reranker is not None and reranker.enabled)
        candidate_k = max(top_k, RERANK_CANDIDATES) if should_rerank else top_k
        candidates = [(entry, score) for score, entry in weighted[:candidate_k]]
        ranked = candidates[:top_k]
        self._last_rerank_summary = ""
        if should_rerank and reranker is not None and len(candidates) > 3:
            result = reranker.curate([entry for entry, _score in candidates], query)
            by_id_rank = {entry.id: (entry, score) for entry, score in candidates}
            ordered: list[tuple[MemoryEntry, float]] = []
            for entry_id in result.selected_ids:
                hit = by_id_rank.get(entry_id)
                if hit is not None:
                    ordered.append(hit)
            if ordered:
                ranked = ordered[:top_k]
            self._last_rerank_summary = result.summary
        touched: set[MemoryScope] = set()
        now = time.time()
        for entry, _score in ranked:
            entry.usage_count += 1
            entry.last_accessed = now
            if entry.tier == MemoryTier.ARCHIVAL:
                entry.tier = MemoryTier.SHORT_TERM
            touched.add(entry.scope)
        for scope in touched:
            self._save_scope(scope)
        return ranked

    def _should_use_graph(
        self,
        query: str,
        bm25_ids: list[str],
        vector_ids: list[str],
        dense_ids: list[str],
    ) -> bool:
        if not self._graph.enabled:
            return False
        if graph_shaped_query(query):
            return True
        seen = set(bm25_ids) | set(vector_ids) | set(dense_ids)
        return len(seen) < 3 and self._graph.edge_count > 0

    def maintain(self, *, session_end: bool = False, now: float | None = None) -> str:
        """Apply tier rules. session_end demotes working notes after a chat."""

        clock = time.time() if now is None else now
        moved = {"short_term": 0, "long_term": 0, "archival": 0}
        dirty: set[MemoryScope] = set()
        for entry in self.all_entries():
            before = entry.tier
            _apply_tier_rules(entry, clock=clock, session_end=session_end)
            if entry.tier != before:
                dirty.add(entry.scope)
                moved[entry.tier.value] = moved.get(entry.tier.value, 0) + 1
        for scope in dirty:
            self._save_scope(scope)
        if not dirty:
            return "Memory tiers unchanged."
        parts = [f"{name}={count}" for name, count in moved.items() if count]
        return "Memory maintain: " + ", ".join(parts)

    def relevant_markdown(self, *, query: str | None = None, top_k: int = DEFAULT_TOP_K, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        query_text = (query or "").strip()
        parts: list[str] = []
        used = 0
        if query_text and sum(len(store.entries) for store in self.stores.values()) > top_k:
            hits = self.search(query_text, top_k=top_k, rerank=True)
            if hits:
                if self._last_rerank_summary:
                    parts.append(self._last_rerank_summary)
                    used += len(self._last_rerank_summary)
                grouped: dict[MemoryScope, list[MemoryEntry]] = {}
                for entry, _score in hits:
                    grouped.setdefault(entry.scope, []).append(entry)
                for scope in (MemoryScope.LOCAL, MemoryScope.PROJECT, MemoryScope.USER):
                    entries = grouped.get(scope)
                    if not entries:
                        continue
                    chunk = MemoryStore(scope=scope, entries=entries).format_prompt()
                    if used + len(chunk) > max_chars:
                        break
                    parts.append(chunk)
                    used += len(chunk)
                return "\n".join(parts).strip()

        for scope in (MemoryScope.LOCAL, MemoryScope.PROJECT, MemoryScope.USER):
            store = self.stores[scope]
            if not store.entries:
                continue
            chunk = store.format_prompt(sort_by_tier=True)
            if used + len(chunk) > max_chars:
                remain = max(0, max_chars - used)
                snippet = chunk[:remain].rstrip() if remain else ""
                if snippet:
                    parts.append(snippet + "\n... (memory truncated)")
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(parts).strip()

    def format_list(self, scope: MemoryScope | None = None) -> str:
        scopes = [scope] if scope else list(MemoryScope)
        lines: list[str] = []
        total = 0
        for current in scopes:
            entries = self.stores[current].entries
            total += len(entries)
            lines.append(f"{current.value}: {len(entries)} entries")
            for entry in entries[:12]:
                preview = entry.prompt_text().replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                extra = f" [{entry.tier.value}]"
                if entry.category != "general":
                    extra += f" [{entry.category}]"
                lines.append(f"  - {preview}{extra}")
        if total == 0:
            return (
                "No project memory yet.\n"
                "Use /memory add [user:|project:|local:] <note> or edit .microcode/MEMORY.md"
            )
        lines.append(f"total: {total}")
        return "\n".join(lines)


def _default_home() -> Path:
    override = (os.environ.get("MICROCODE_HOME") or "").strip()
    if override:
        return Path(override)
    return Path.home() / ".microcode"


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as writer:
            writer.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _bm25_rank(entries: list[MemoryEntry], query: str) -> list[str]:
    query_tokens = tokenize(query, drop_stopwords=True)
    if not query_tokens:
        return []
    docs = [tokenize(f"{entry.content} {entry.category} {' '.join(entry.tags)}") for entry in entries]
    idf = compute_idf(docs)
    avgdl = compute_avgdl(docs)
    scored: list[tuple[float, str]] = []
    query_lower = query.lower()
    for entry, tokens in zip(entries, docs):
        score = bm25_score(query_tokens, tokens, idf, avgdl)
        content_lower = entry.content.lower()
        if query_lower in content_lower:
            score += 2.0
        elif any(token in content_lower for token in query_tokens):
            score += 1.0
        if any(tag.lower() == query_lower for tag in entry.tags):
            score += 5.0
        if query_lower in entry.category.lower():
            score += 1.0
        if score > 0:
            scored.append((score, entry.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry_id for _score, entry_id in scored]


def _archive(entry: MemoryEntry) -> None:
    entry.tier = MemoryTier.ARCHIVAL
    if not entry.summary:
        entry.summary = summarize_text(entry.content)


def _apply_tier_rules(entry: MemoryEntry, *, clock: float, session_end: bool) -> None:
    if entry.tier == MemoryTier.WORKING:
        if session_end or clock - entry.created_at >= WORKING_SECONDS:
            entry.tier = MemoryTier.SHORT_TERM
    if entry.tier == MemoryTier.SHORT_TERM:
        age = clock - entry.created_at
        idle = clock - entry.last_accessed
        if entry.usage_count >= PROMOTE_USES and age >= PROMOTE_AGE_SECONDS:
            entry.tier = MemoryTier.LONG_TERM
            return
        if idle >= ARCHIVE_SECONDS and entry.usage_count < PROMOTE_USES:
            _archive(entry)
            return
    if entry.tier == MemoryTier.LONG_TERM:
        if clock - entry.last_accessed >= ARCHIVE_SECONDS:
            _archive(entry)


class ProjectMemory:
    """Workspace-facing memory: project markdown plus user/local JSON scopes."""

    def __init__(
        self,
        workspace: str | Path,
        max_chars: int = DEFAULT_MAX_CHARS,
        *,
        home: str | Path | None = None,
        dense: DenseVectorStore | None = None,
        reranker: MemoryReranker | None = None,
        graph: MemoryGraphStore | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.max_chars = max_chars
        self.manager = MemoryManager(
            self.workspace, home=home, dense=dense, reranker=reranker, graph=graph
        )
        self.path = self.manager.markdown_path(MemoryScope.PROJECT)

    def read(self) -> str:
        if not self.manager.stores[MemoryScope.PROJECT].entries and not self.path.is_file():
            return ""
        if self.path.is_file():
            try:
                return self.path.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return self.manager.stores[MemoryScope.PROJECT].format_markdown().strip()

    def display(self) -> str:
        listing = self.manager.format_list()
        text = self.read()
        if not text and listing.startswith("No project memory"):
            return listing
        if text:
            return f"{self.path}\n\n{text}\n\n{listing}"
        return listing

    def append(self, note: str, *, scope: MemoryScope | None = None) -> str:
        cleaned = note.strip()
        if not cleaned:
            return "Usage: /memory add [user:|project:|local:] <note>"
        chosen = scope or MemoryScope.PROJECT
        match = SCOPE_PREFIX.match(cleaned)
        if match:
            chosen = MemoryScope(match.group(1).lower())
            cleaned = match.group(2).strip()
        if not cleaned:
            return "Usage: /memory add [user:|project:|local:] <note>"
        if cleaned.startswith(("- ", "* ")):
            cleaned = cleaned[2:].strip()
        entry = self.manager.add(cleaned, scope=chosen)
        return f"Remembered ({entry.scope.value}/{entry.tier.value}): - {entry.content}"

    def notes(self) -> list[MemoryNote]:
        return [entry.as_note() for entry in self.manager.stores[MemoryScope.PROJECT].entries]

    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[tuple[MemoryNote, float]]:
        hits = self.manager.search(query, top_k=top_k)
        return [(entry.as_note(), score) for entry, score in hits]

    def format_search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> str:
        cleaned = query.strip()
        if not cleaned:
            return "Usage: /memory search <query>"
        hits = self.manager.search(cleaned, top_k=top_k)
        if not hits:
            return "No matching memory notes."
        return "\n".join(
            f"{score:.3f}  [{entry.scope.value}/{entry.tier.value}] {entry.prompt_text()}"
            for entry, score in hits
        )

    def maintain(self, *, session_end: bool = False, now: float | None = None) -> str:
        return self.manager.maintain(session_end=session_end, now=now)

    def inject(self, system_prompt: str, *, query: str | None = None, top_k: int = DEFAULT_TOP_K) -> str:
        total = len(self.manager.all_entries())
        if total == 0:
            return system_prompt
        retrieved = bool(query and query.strip() and total > top_k)
        body = self.manager.relevant_markdown(query=query, top_k=top_k, max_chars=self.max_chars)
        if not body:
            return system_prompt
        if not retrieved and len(body) > self.max_chars:
            body = body[: self.max_chars].rstrip() + "\n... (memory truncated)"
        if retrieved:
            blurb = (
                "The following notes were retrieved for this turn from project memory. "
                "Follow them unless the user says otherwise."
            )
        else:
            blurb = (
                "The following notes persist across sessions. "
                "Follow them unless the user says otherwise."
            )
        suffix = f"{MEMORY_HEADING}\n\n{blurb}\n\n{body}"
        return f"{system_prompt.rstrip()}\n\n{suffix}"


def apply_system_message(messages: list[ChatMessage], content: str) -> list[ChatMessage]:
    """Replace or insert the leading system message. Leaves other rows unchanged."""

    system = {"role": "system", "content": content}
    next_messages = list(messages)
    if next_messages and next_messages[0].get("role") == "system":
        next_messages[0] = system
        return next_messages
    return [system, *next_messages]


def parse_memory_command(text: str) -> tuple[str, str]:
    """Return ('show'|'add'|'search'|'maintain'|'help', payload) for a /memory line."""

    rest = text.strip()[len("/memory") :].strip()
    if not rest:
        return "show", ""
    lowered = rest.lower()
    if lowered == "add":
        return "add", ""
    if lowered.startswith("add "):
        return "add", rest[4:].strip()
    if lowered == "search":
        return "search", ""
    if lowered.startswith("search "):
        return "search", rest[7:].strip()
    if lowered == "maintain":
        return "maintain", ""
    return "help", ""
