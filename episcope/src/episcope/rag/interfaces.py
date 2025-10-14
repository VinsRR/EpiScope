"""Abstract interfaces for core RAG components.

This module defines abstract base classes (ABCs) for the major phases
of retrieval‑augmented generation (RAG) in the EpiScope project.  By
defining these interfaces, implementations of indexing, retrieval and
generation can be swapped out or extended without modifying calling
code.  The abstract classes follow a minimal, Pythonic API that
captures the required behaviours.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Dict, Optional, Any


class AbstractIndexer(ABC):
    """Abstract base class for document indexers.

    Concrete implementations encapsulate the logic for building an
    index over a collection of documents and performing similarity
    search.  Indexers may target different backends (in‑memory,
    FAISS, Qdrant, etc.) and support different document granularities
    (global vs. per‑paper indexing).
    """

    @abstractmethod
    def index_documents(self, docs: Iterable[Dict[str, Any]], *, namespace: Optional[str] = None) -> None:
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



class AbstractRetriever(ABC):
    """Abstract base class for retrieval strategies.

    Retrievers wrap an underlying indexer (or multiple indexers)
    together with optional query augmentation mechanisms such as
    HYDE.  The ``retrieve`` method returns a list of context
    dictionaries suitable for downstream answer generation.  More
    specialised retrievers may expose additional methods (e.g. for
    multi‑paper retrieval).  Clients should depend on the abstract
    class rather than concrete implementations to allow swapping
    implementations.
    """

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> Sequence[Dict[str, Any]]:
        """Retrieve contexts for a query.

        Args:
            query: The user query.
            top_k: Maximum number of contexts to return.
            **kwargs: Additional arguments for the retrieval strategy.

        Returns:
            A sequence of context dictionaries containing text and
            associated metadata.  The exact fields depend on the
            implementation.  At minimum, each context should contain
            a ``content`` key with the text snippet.
        """


class AbstractGenerator(ABC):
    """Abstract base class for answer generation.

    Generators combine retrieved contexts with a question to produce
    an answer and optional provenance metadata.  The answer format
    should be compatible with both API/CLI outputs and interactive
    UIs.
    """

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[Dict[str, Any]], **kwargs: Any) -> Any:
        """Generate an answer given a question and supporting contexts.

        Args:
            question: The question to answer.
            contexts: A sequence of context dictionaries returned by
                a retriever.  Each context should at least contain a
                ``content`` field with the text.
            **kwargs: Additional parameters (e.g. model hints).

        Returns:
            Implementation‑specific answer object (e.g. a Provenance
            instance).
        """