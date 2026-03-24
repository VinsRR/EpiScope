from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class Embedder(ABC):
    """Abstract base class for dense single-vector embedders."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the embedding model."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the dense embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings."""

    def embed_units(self, units: Iterable[Dict[str, str]]) -> List[List[float]]:
        """Embed the ``content`` field of each unit in ``units``."""
        return self.embed_texts(u["content"] for u in units)


class SparseEmbedder(ABC):
    """Abstract base class for sparse (SPLADE-style) embedders.

    Each embedding is a dict of non-zero vocabulary indices to weights,
    rather than a dense fixed-length vector.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the sparse embedding model."""

    @abstractmethod
    def embed_text(self, text: str) -> Dict[str, Any]:
        """Embed a single text string.

        Returns a dict with at minimum:
            ``indices``: List[int] — non-zero vocabulary token ids
            ``values``:  List[float] — corresponding weights
        """

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        """Embed an iterable of text strings."""

    def embed_units(self, units: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Embed the ``content`` field of each unit in ``units``."""
        return self.embed_texts(u["content"] for u in units)


class LateEmbedder(ABC):
    """Abstract base class for late-interaction (ColBERT-style) embedders.

    Unlike dense embedders, late interaction models produce a matrix of
    per-token vectors per text rather than a single pooled vector.
    They also require separate encoding paths for queries and documents,
    since many models apply different prefixes or masking strategies to each.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the late-interaction model."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """The token embedding dimension."""

    @abstractmethod
    def embed_query(self, text: str) -> List[List[float]]:
        """Encode a query string into per-token vectors. Shape: [T, D]."""

    @abstractmethod
    def embed_queries(self, texts: Iterable[str]) -> List[List[List[float]]]:
        """Encode an iterable of query strings. Shape per item: [T, D]."""

    @abstractmethod
    def embed_document(self, text: str) -> List[List[float]]:
        """Encode a document string into per-token vectors. Shape: [T, D]."""

    @abstractmethod
    def embed_documents(self, texts: Iterable[str]) -> List[List[List[float]]]:
        """Encode an iterable of document strings. Shape per item: [T, D]."""

    def embed_text(self, text: str) -> List[List[float]]:
        """Backward-compatible alias for embed_document."""
        return self.embed_document(text)

    def embed_texts(self, texts: Iterable[str]) -> List[List[List[float]]]:
        """Backward-compatible alias for embed_documents."""
        return self.embed_documents(texts)