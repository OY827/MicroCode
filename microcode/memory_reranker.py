from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RERANK_CANDIDATES = 15
RERANK_KEEP = 5
RERANK_CONTENT_CHARS = 200
RERANK_CACHE_SIZE = 256
RERANK_CACHE_TTL = 60.0

RERANK_PROMPT = """You filter project-memory search hits. Keep only notes that help the current task.

Task: {task}

Candidates:
{candidates}

Select the most relevant notes (at most {keep}). Reject cross-topic noise.
Mark contradictory pairs if any.
Write a 2-3 sentence context summary.

Return ONLY JSON:
{{"selected": ["id1"], "rejected": [{{"id": "id2", "reason": "..."}}], "conflicts": [], "summary": "..."}}"""


def rerank_disabled() -> bool:
    raw = (os.environ.get("MICROCODE_MEMORY_RERANK") or "").strip().lower()
    return raw in {"0", "false", "no", "off"}


@dataclass
class RerankResult:
    selected_ids: list[str]
    rejected: list[dict[str, str]] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.7

    @classmethod
    def fallback(cls, candidate_ids: list[str], *, keep: int = RERANK_KEEP) -> RerankResult:
        chosen = candidate_ids[: min(keep, len(candidate_ids))]
        return cls(selected_ids=chosen, confidence=0.3)


def parse_rerank_json(text: str, valid_ids: set[str]) -> RerankResult | None:
    """Parse a model reply into selected ids. None means the payload was unusable."""

    json_text = text.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:])
        if json_text.endswith("```"):
            json_text = json_text[:-3].strip()
    start = json_text.find("{")
    end = json_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(json_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw_selected = data.get("selected") or []
    if not isinstance(raw_selected, list):
        return None
    selected = [item for item in raw_selected if isinstance(item, str) and item in valid_ids]
    rejected = data.get("rejected") if isinstance(data.get("rejected"), list) else []
    conflicts = data.get("conflicts") if isinstance(data.get("conflicts"), list) else []
    summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
    return RerankResult(
        selected_ids=selected,
        rejected=[item for item in rejected if isinstance(item, dict)],
        conflicts=[item for item in conflicts if isinstance(item, dict)],
        summary=summary.strip(),
        confidence=0.7 if selected else 0.3,
    )


def complete_from_adapter(model: Any, prompt: str) -> str:
    """Ask a ModelAdapter for text only: no tools, no stream, isolated messages."""

    from microcode.tooling import ToolRegistry

    saved_tools = getattr(model, "tools", None)
    saved_stream = getattr(model, "stream", None)
    try:
        if saved_tools is not None:
            model.tools = ToolRegistry([])
        if saved_stream is not None:
            model.stream = False
        step = model.next(
            [
                {"role": "system", "content": "Reply with JSON only. Do not call tools."},
                {"role": "user", "content": prompt},
            ]
        )
    finally:
        if saved_tools is not None:
            model.tools = saved_tools
        if saved_stream is not None:
            model.stream = saved_stream
    if getattr(step, "type", "") != "assistant":
        raise RuntimeError("rerank model returned tool calls")
    content = str(getattr(step, "content", "") or "").strip()
    if not content:
        raise RuntimeError("rerank model returned empty text")
    return content


class MemoryReranker:
    """LLM pass over fused search hits. Any failure keeps the original ranking."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        *,
        max_candidates: int = RERANK_CANDIDATES,
        keep: int = RERANK_KEEP,
        cache_ttl: float = RERANK_CACHE_TTL,
    ) -> None:
        self._complete = complete
        self._max_candidates = max_candidates
        self._keep = keep
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, RerankResult]] = {}
        self.call_count = 0
        self.cache_hits = 0
        self.fallback_count = 0

    @property
    def enabled(self) -> bool:
        return self._complete is not None and not rerank_disabled()

    @classmethod
    def from_model(cls, model: Any | None) -> MemoryReranker | None:
        if model is None or rerank_disabled():
            return None
        if type(model).__name__ == "MockModelAdapter":
            return None
        return cls(complete=lambda prompt: complete_from_adapter(model, prompt))

    def curate(self, candidates: list[Any], task: str) -> RerankResult:
        ids = [str(getattr(item, "id", "")) for item in candidates if getattr(item, "id", "")]
        if not self.enabled or not ids:
            return RerankResult.fallback(ids, keep=self._keep)
        pool = candidates[: self._max_candidates]
        pool_ids = [str(getattr(item, "id", "")) for item in pool]
        cache_key = hashlib.md5(
            (task[:100] + "|" + "|".join(pool_ids)).encode(),
            usedforsecurity=False,
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            stamp, result = cached
            if time.time() - stamp < self._cache_ttl:
                self.cache_hits += 1
                return result
            del self._cache[cache_key]
        self.call_count += 1
        try:
            result = self._call(pool, task)
        except Exception:
            self.fallback_count += 1
            return RerankResult.fallback(pool_ids, keep=self._keep)
        if not result.selected_ids:
            result = RerankResult(
                selected_ids=pool_ids[: min(3, len(pool_ids))],
                rejected=result.rejected,
                conflicts=result.conflicts,
                summary=result.summary,
                confidence=0.3,
            )
        self._store(cache_key, result)
        return result

    def _call(self, candidates: list[Any], task: str) -> RerankResult:
        if self._complete is None:
            raise RuntimeError("reranker has no complete function")
        lines: list[str] = []
        valid: set[str] = set()
        for item in candidates:
            entry_id = str(getattr(item, "id", ""))
            if not entry_id:
                continue
            valid.add(entry_id)
            content = str(getattr(item, "content", "")).replace("\n", " ")[:RERANK_CONTENT_CHARS]
            category = str(getattr(item, "category", "") or "general")
            tags = ", ".join(str(tag) for tag in (getattr(item, "tags", None) or [])[:5])
            usage = int(getattr(item, "usage_count", 0) or 0)
            extra = f" tags={tags}" if tags else ""
            lines.append(f"  [{entry_id}] category={category} used={usage}x{extra}\n    {content}")
        prompt = RERANK_PROMPT.format(
            task=task.strip()[:300],
            candidates="\n".join(lines),
            keep=self._keep,
        )
        parsed = parse_rerank_json(self._complete(prompt), valid)
        if parsed is None:
            raise RuntimeError("rerank JSON parse failed")
        return parsed

    def _store(self, key: str, result: RerankResult) -> None:
        if len(self._cache) >= RERANK_CACHE_SIZE:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            del self._cache[oldest]
        self._cache[key] = (time.time(), result)
