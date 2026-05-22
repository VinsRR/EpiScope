"""Abstract interfaces for core RAG components.

This module defines abstract base classes (ABCs) for indexing and
generation. Retrieval now lives under ``episcope.rag.retrieval`` with a
single shared ``BaseRetriever`` foundation for vector-database-backed
retrieval pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Optional, Any

from episcope.rag.generation.base import Generator


class AbstractIndexer(ABC):
    """Abstract base class for document indexers.

    Concrete implementations encapsulate the logic for building an
    index over a collection of documents and performing similarity
    search.  Indexers may target different backends (in‑memory,
    FAISS, Qdrant, etc.) and support different document granularities
    (global vs. per‑paper indexing).
    """

    @abstractmethod
    def index_documents(
        self, docs: Iterable[Dict[str, Any]], *, namespace: Optional[str] = None
    ) -> None:
        """Index the given documents.

        Each document is a mapping with at least a ``content`` field
        containing the text to be embedded.  Implementations may
        accept additional metadata fields which are stored as payloads.

        Args:
            docs: An iterable of document dictionaries.  Each dict
                should contain a ``content`` key with the text to
                index.  Additional keys may be used to store
                metadata.
            namespace: Optional logical grouping for the documents
                (e.g. per‑project or per‑paper).  Some backends
                support logical isolation of vectors; others may
                ignore this parameter.
        """


class AbstractGenerator(Generator):
    """Backward-compatible alias for the canonical generation contract."""
