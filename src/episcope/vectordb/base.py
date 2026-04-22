"""
Abstract interface for a Vector Database.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

class AbstractVectorDB(ABC):
    """Abstract base class for a vector database."""

    @abstractmethod
    def upsert(
        self,
        points: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
        embed_models: Optional[Dict[str, str]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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
    def get_embedding_model(self) -> Optional[Union[str, Dict[str, str]]]:
        """Get the name of the embedding model used for the database."""
        pass

    @abstractmethod
    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        """Get the chunking configuration used for the database."""
        pass

    def capabilities(self) -> Dict[str, bool]:
        """Report which retrieval modalities the DB supports.

        Dense-only databases can inherit this default implementation.
        Backends with sparse or late-interaction support should override it.
        """
        return {
            "dense": True,
            "sparse": False,
            "late": False,
        }

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Compatibility alias for dense search on dense-only backends."""
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )

    def search_sparse(
        self,
        query_sparse: Dict[str, List[float]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError("This vector database does not support sparse search.")

    def search_hybrid(
        self,
        dense_query: List[float],
        sparse_query: Dict[str, List[float]],
        top_k: int,
        prefetch_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError("This vector database does not support hybrid search.")

    def rerank_late(
        self,
        query_late: List[List[float]],
        candidate_ids: List[Any],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError("This vector database does not support late-interaction reranking.")
