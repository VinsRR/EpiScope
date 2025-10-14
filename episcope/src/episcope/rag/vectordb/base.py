"""
Abstract interface for a Vector Database.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

class AbstractVectorDB(ABC):
    """Abstract base class for a vector database."""

    @abstractmethod
    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        """Upsert points into the vector database."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Search the vector database."""
        pass
