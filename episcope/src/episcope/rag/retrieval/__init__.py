"""
retrieve package

This package contains implementations of the retrieval components used
by EpiScope.  The subpackages under ``retrieve/`` implement specific
retrieval strategies such as text‑only RAG, multimodal RAG and graph
RAG.  The top‑level ``RAGFactory`` class can be used to instantiate a
retrieval strategy by name.

The import of the heavy RAG factory is wrapped in a try/except so that
environments without optional dependencies (such as ``qdrant_client``)
can still import this module.  When dependencies are missing, a dummy
``RAGFactory`` class is defined to avoid ``ImportError`` during test
discovery.  This stub raises on use to indicate that the real
implementation is unavailable.
"""
from __future__ import annotations
from typing import Any

from episcope.rag.retrieval.base import BaseRetriever

__all__ = [
    "BaseRetriever",
    "Retriever",
    "SemanticCandidateRetriever",
    "HybridCandidateRetriever",
    "SparseCandidateRetriever",
    "RAGFactory",
]


class _UnavailableRetriever:  # type: ignore
    """Fallback used when optional retrieval dependencies are missing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "Retriever components are unavailable because required retrieval dependencies are missing."
        )


try:
    from episcope.rag.retrieval.candidates import (
        HybridCandidateRetriever,
        SemanticCandidateRetriever,
        SparseCandidateRetriever,
    )
    from episcope.rag.retrieval.retriever import Retriever
except Exception:
    Retriever = _UnavailableRetriever
    HybridCandidateRetriever = _UnavailableRetriever
    SemanticCandidateRetriever = _UnavailableRetriever
    SparseCandidateRetriever = _UnavailableRetriever

try:
    # Attempt to import the concrete RAGFactory implementation.  This
    # import may fail if optional dependencies like qdrant_client or
    # fastembed are not installed.  In that case, we provide a stub
    # class instead.
    from episcope.rag.factory import RAGFactory as _ConcreteRAGFactory
    RAGFactory = _ConcreteRAGFactory
except Exception:
    class RAGFactory:  # type: ignore
        """Fallback RAGFactory used when optional dependencies are missing."""
        @classmethod
        def get(cls, *args: Any, **kwargs: Any) -> Any:
            """Raise ImportError since no RAG implementations are available."""
            raise ImportError(
                "RAGFactory is unavailable because required retrieval dependencies are missing."
            )

        def __getattr__(self, name: str) -> None:
            raise ImportError(
                "RAGFactory is unavailable because required retrieval dependencies are missing."
            )
