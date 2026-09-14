from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class QueryTransformer(ABC):
    """Transforms a query before candidate retrieval."""

    @abstractmethod
    def transform(self, query: str) -> str:
        """Return the transformed query text."""


class CandidateRetriever(ABC):
    """Low-level first-stage retriever for one retrieval signal."""

    source = "candidate"

    @abstractmethod
    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw candidate payloads from one retrieval signal."""


class FusionStrategy(ABC):
    """Combines multiple candidate lists into a single ranking."""

    @abstractmethod
    def fuse(
        self,
        results_lists: Sequence[Sequence[Dict[str, Any]]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked candidate lists."""


class CandidateReranker(ABC):
    """Reranks raw candidates before final SearchResult conversion."""

    @abstractmethod
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank and return a new candidate list."""
