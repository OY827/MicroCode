from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

WORD_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "how",
        "what",
        "should",
        "please",
        "的",
        "了",
        "吗",
        "呢",
        "怎么",
        "如何",
        "一下",
    }
)
SIMILARITY_FLOOR = 0.08
DENSE_SIMILARITY_FLOOR = 0.3
DENSE_ENCODE_CHARS = 500
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
_UNSET = object()
_default_dense_encoder: object = _UNSET

CLASSIFICATION_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("architecture", ("architecture", "design", "pattern", "api", "rest", "backend", "架构", "设计"), "design-pattern"),
    ("testing", ("test", "assert", "pytest", "unit", "测试"), "test"),
    ("configuration", ("config", "settings", "env", "配置", "环境"), "config"),
    ("workflow", ("git", "commit", "branch", "merge", "工作流"), "git"),
    ("security", ("security", "auth", "permission", "安全", "权限"), "security"),
    ("performance", ("performance", "optimization", "性能", "优化"), "optimization"),
    ("convention", ("convention", "style", "naming", "规范", "风格"), "style"),
    ("code-pattern", ("function", "method", "class", "函数", "方法"), "function"),
)


@dataclass(frozen=True)
class MemoryNote:
    id: str
    content: str


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Split English words and CJK characters/bigrams for TF-IDF."""

    tokens = [word.lower() for word in WORD_RE.findall(text)]
    chars = CJK_RE.findall(text)
    tokens.extend(chars)
    tokens.extend(chars[index] + chars[index + 1] for index in range(len(chars) - 1))
    if drop_stopwords:
        tokens = [token for token in tokens if token not in QUERY_STOPWORDS]
    return tokens


def parse_notes(markdown: str) -> list[MemoryNote]:
    """Turn MEMORY.md into one note per bullet or paragraph."""

    notes: list[MemoryNote] = []
    paragraph: list[str] = []

    def flush() -> None:
        text = " ".join(paragraph).strip()
        paragraph.clear()
        if text:
            notes.append(MemoryNote(id=f"n{len(notes) + 1}", content=text))

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            continue
        if line.startswith(("- ", "* ")):
            flush()
            notes.append(MemoryNote(id=f"n{len(notes) + 1}", content=line[2:].strip()))
            continue
        paragraph.append(line)
    flush()
    return notes


class SparseVectorStore:
    """In-memory TF-IDF vectors. Cosine similarity, no extra packages."""

    def __init__(self) -> None:
        self._notes: dict[str, MemoryNote] = {}
        self._doc_vectors: dict[str, dict[int, float]] = {}
        self._term_to_id: dict[str, int] = {}
        self._doc_freq: dict[int, int] = {}
        self._doc_count = 0

    def clear(self) -> None:
        self._notes.clear()
        self._doc_vectors.clear()
        self._term_to_id.clear()
        self._doc_freq.clear()
        self._doc_count = 0

    def index(self, notes: list[MemoryNote]) -> int:
        self.clear()
        usable = [note for note in notes if note.content.strip()]
        counts = [Counter(tokenize(note.content)) for note in usable]
        for bag in counts:
            for term in bag:
                if term not in self._term_to_id:
                    self._term_to_id[term] = len(self._term_to_id)
        self._doc_count = len(usable)
        for bag in counts:
            seen = {self._term_to_id[term] for term in bag}
            for term_id in seen:
                self._doc_freq[term_id] = self._doc_freq.get(term_id, 0) + 1

        for note, bag in zip(usable, counts):
            total = max(sum(bag.values()), 1)
            vector: dict[int, float] = {}
            for term, freq in bag.items():
                term_id = self._term_to_id[term]
                tf = freq / total
                df = self._doc_freq.get(term_id, 1)
                idf = math.log((self._doc_count + 1) / (df + 1)) + 1
                vector[term_id] = tf * idf
            self._notes[note.id] = note
            self._doc_vectors[note.id] = vector
        return len(self._notes)

    def search(self, query: str, *, top_k: int = 8) -> list[tuple[MemoryNote, float]]:
        query_tokens = tokenize(query, drop_stopwords=True)
        if not query_tokens or not self._doc_vectors:
            return []
        bag = Counter(query_tokens)
        total = max(sum(bag.values()), 1)
        query_vec: dict[int, float] = {}
        for term, freq in bag.items():
            term_id = self._term_to_id.get(term)
            if term_id is None:
                continue
            df = self._doc_freq.get(term_id, 1)
            idf = math.log((self._doc_count + 1) / (df + 1)) + 1
            query_vec[term_id] = (freq / total) * idf
        if not query_vec:
            return []
        query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values()))
        if query_norm == 0:
            return []

        hits: list[tuple[MemoryNote, float]] = []
        for note_id, doc_vec in self._doc_vectors.items():
            doc_norm = math.sqrt(sum(weight * weight for weight in doc_vec.values()))
            if doc_norm == 0:
                continue
            dot = sum(query_vec.get(term_id, 0.0) * weight for term_id, weight in doc_vec.items())
            score = dot / (query_norm * doc_norm)
            if score > SIMILARITY_FLOOR:
                hits.append((self._notes[note_id], score))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:top_k]


Encoder = Callable[[str], list[float] | object]


def dense_memory_disabled() -> bool:
    raw = (os.environ.get("MICROCODE_DENSE_MEMORY") or "").strip().lower()
    return raw in {"0", "false", "no", "off"}


def _as_floats(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _cosine_dense(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _load_minilm_encoder() -> Encoder | None:
    """Load all-MiniLM-L6-v2 when sentence-transformers is installed."""

    if dense_memory_disabled():
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer(DENSE_MODEL_NAME)

        def encode(text: str) -> list[float]:
            return _as_floats(model.encode(text, show_progress_bar=False))

        return encode
    except Exception:
        return None


def reset_default_dense_encoder() -> None:
    global _default_dense_encoder
    _default_dense_encoder = _UNSET


def default_dense_encoder() -> Encoder | None:
    global _default_dense_encoder
    if _default_dense_encoder is _UNSET:
        _default_dense_encoder = _load_minilm_encoder()
    return _default_dense_encoder  # type: ignore[return-value]


class DenseVectorStore:
    """Optional MiniLM embeddings. Disabled unless an encoder is injected or installed."""

    def __init__(self, encoder: Encoder | None = None, *, load_default: bool = True) -> None:
        self._encoder = encoder if encoder is not None else (default_dense_encoder() if load_default else None)
        self._notes: dict[str, MemoryNote] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._fingerprints: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self._encoder is not None

    def clear(self) -> None:
        self._notes.clear()
        self._embeddings.clear()
        self._fingerprints.clear()

    def forget_except(self, live_ids: set[str]) -> None:
        stale = [note_id for note_id in self._embeddings if note_id not in live_ids]
        for note_id in stale:
            self._notes.pop(note_id, None)
            self._embeddings.pop(note_id, None)
            self._fingerprints.pop(note_id, None)

    def index(self, notes: list[MemoryNote]) -> int:
        if not self.enabled or self._encoder is None:
            return 0
        added = 0
        for note in notes:
            snippet = note.content.strip()[:DENSE_ENCODE_CHARS]
            if not snippet:
                continue
            if self._fingerprints.get(note.id) == snippet:
                self._notes[note.id] = note
                continue
            try:
                vector = _as_floats(self._encoder(snippet))
            except Exception:
                continue
            if not vector:
                continue
            self._notes[note.id] = note
            self._embeddings[note.id] = vector
            self._fingerprints[note.id] = snippet
            added += 1
        return added

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        candidate_ids: set[str] | None = None,
    ) -> list[tuple[MemoryNote, float]]:
        if not self.enabled or self._encoder is None or not self._embeddings:
            return []
        snippet = query.strip()[:DENSE_ENCODE_CHARS]
        if not snippet:
            return []
        try:
            query_vec = _as_floats(self._encoder(snippet))
        except Exception:
            return []
        if not query_vec:
            return []
        hits: list[tuple[MemoryNote, float]] = []
        for note_id, vector in self._embeddings.items():
            if candidate_ids is not None and note_id not in candidate_ids:
                continue
            note = self._notes.get(note_id)
            if note is None:
                continue
            score = _cosine_dense(query_vec, vector)
            if score > DENSE_SIMILARITY_FLOOR:
                hits.append((note, score))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:top_k]


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for tokens in documents:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {term: math.log((n + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}


def compute_avgdl(documents: list[list[str]]) -> float:
    if not documents:
        return 0.0
    return sum(len(doc) for doc in documents) / len(documents)


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avgdl: float,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    if not query_tokens or not doc_tokens or avgdl == 0:
        return 0.0
    tf_doc = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term not in idf:
            continue
        tf = float(tf_doc.get(term, 0))
        if tf == 0:
            continue
        denominator = tf + k1 * (1 - b + b * (len(doc_tokens) / avgdl))
        score += idf[term] * (tf * (k1 + 1) / denominator)
    return score


def reciprocal_rank_fusion(*ranked_ids: list[str], k: int = RRF_K) -> list[str]:
    """Merge ranked id lists. MiniCode uses the same fusion for BM25 + vectors."""

    scores: dict[str, float] = {}
    for ranking in ranked_ids:
        for index, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + index + 1)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def classify_content(content: str) -> tuple[str, list[str]]:
    lowered = content.lower()
    best = ""
    best_score = 0
    tags: list[str] = []
    for category, keywords, tag in CLASSIFICATION_RULES:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best = category
            tags = [tag]
    if best_score == 0:
        return "general", []
    return best, tags
