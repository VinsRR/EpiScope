"""
core/rag.py

Definitions of abstract base classes for retrieval-augmented generation.

Retrieval-augmented generation (RAG) methods follow a common lifecycle: they
accept documents to index, allow retrieval of relevant contexts for a query,
and generate a final answer from those contexts using an LLM.  Concrete
implementations may differ in their embedding models, index backends or
prompt construction, but they all conform to the same interface.  See
``retrieve/rag/`` for concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence


class AbstractRAG(ABC):
    """Abstract base class for retrieval‑augmented generation methods.

    Subclasses must implement methods to index documents, retrieve
    contexts and generate answers.  The ``run`` method orchestrates
    retrieval and generation and is provided as a convenience.
    """

    @abstractmethod
    def index(self, source: str | Iterable[str]) -> None:
        """Index one or more documents.

        ``source`` may be a path to a PDF file or directory, or an
        iterable of textual content.  Implementations should be
        flexible to support both file system and in‑memory ingestion.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Any]:
        """Retrieve the top‑``k`` contexts relevant to ``query``.

        Returns an ordered list of objects representing the retrieved
        contexts.  The precise type and contents of each entry is
        implementation specific.
        """

    @abstractmethod
    def generate(self, query: str, contexts: Sequence[Any]) -> str:
        """Generate an answer given a query and its retrieved contexts."""

    def run(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        """Convenience method that performs retrieval and generation."""
        contexts = self.retrieve(query, top_k=top_k, **kwargs)
        return self.generate(query, contexts)

