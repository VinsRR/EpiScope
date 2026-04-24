from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from episcope.schemas import SearchResult
from episcope.vectordb.base import AbstractVectorDB


class BaseRetriever(ABC):
    """Reusable retriever plumbing for vector-database-backed strategies."""

    default_source = "retrieval"

    def __init__(self, vectordb: AbstractVectorDB):
        self.vectordb = vectordb
        self._allowed_filter_keys = self._load_allowed_filter_keys()

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """Retrieve contexts for a query."""

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        final_filter = dict(filter or {})
        final_filter["paper_id"] = paper_id
        return self.retrieve(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=final_filter,
        )

    def _load_allowed_filter_keys(self) -> set[str]:
        if hasattr(self.vectordb, "get_payload_keys"):
            return set(self.vectordb.get_payload_keys())
        return set()

    def _prepare_filter(
        self,
        filter: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        final_filter = dict(filter or {})
        namespace = final_filter.pop("paper_id", None) or final_filter.pop(
            "namespace", None
        )

        if self._allowed_filter_keys:
            invalid_keys = sorted(set(final_filter) - self._allowed_filter_keys)
            if invalid_keys:
                raise ValueError(
                    f"Invalid filter key(s): {invalid_keys}. Allowed keys are: {sorted(self._allowed_filter_keys)}"
                )

        return namespace, final_filter

    def _to_search_result(
        self,
        chunk: Dict[str, Any],
        *,
        source: Optional[str] = None,
        rank_score: Optional[float] = None,
    ) -> SearchResult:
        score = float(chunk.get("score", 0.0))
        known_keys = {
            "id",
            "paper_id",
            "text",
            "section_type",
            "section_title",
            "title",
            "is_metadata",
            "score",
            "rrf_score",
            "rank_score",
        }
        return SearchResult(
            id=str(chunk.get("id", "")),
            paper_id=chunk.get("paper_id", ""),
            text=chunk.get("text", ""),
            section_type=chunk.get("section_type", "other"),
            section_title=chunk.get("section_title", chunk.get("title", "")),
            title=chunk.get("title", chunk.get("section_title", "")),
            is_metadata=bool(chunk.get("is_metadata", False)),
            similarity_score=score,
            rank_score=score if rank_score is None else float(rank_score),
            source=source or self.default_source,
            artifacts={
                key: value for key, value in chunk.items() if key not in known_keys
            },
        )

    def _to_search_results(
        self,
        chunks: Iterable[Dict[str, Any]],
        *,
        source: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        for chunk in chunks:
            score = float(chunk.get("score", 0.0))
            if score < similarity_threshold:
                continue

            results.append(
                self._to_search_result(
                    chunk,
                    source=source,
                    rank_score=chunk.get("rrf_score", chunk.get("rank_score")),
                )
            )
        return results
