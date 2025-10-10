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

__all__ = ["RAGFactory"]

try:
    from . import embeddings
    # Attempt to import the concrete RAGFactory implementation.  This
    # import may fail if optional dependencies like qdrant_client or
    # fastembed are not installed.  In that case, we provide a stub
    # class instead.
    from .rag.factory import RAGFactory as _ConcreteRAGFactory  # type: ignore
    RAGFactory = _ConcreteRAGFactory  # type: ignore
except Exception:
    class RAGFactory:  # type: ignore
        """Fallback RAGFactory used when optional dependencies are missing.

        This stubbed factory exposes the same API surface as the
        real implementation but will raise ``ImportError`` on use.
        Having a concrete ``get`` classmethod allows tests to patch
        this method without failing attribute lookups.  Attempting to
        use the fallback factory without patching will result in an
        informative ImportError.
        """

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
