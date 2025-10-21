"""
Abstract interface for a Vector Database.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

class AbstractVectorDB(ABC):
    """Abstract base class for a vector database."""

    @abstractmethod
    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None, chunking_config: Optional[Dict[str, Any]] = None) -> None:
        """Upsert points into the vector database."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Search the vector database with an optional filter."""
        pass

    @abstractmethod
    def get_points(self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        pass

    @abstractmethod
    def get_payload_keys(self) -> set[str]:
        """Get the set of all available payload keys."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Optional[str]:
        """Get the name of the embedding model used for the database."""
        pass

    @abstractmethod
    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        """Get the chunking configuration used for the database."""
        pass