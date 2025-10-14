"""
Wrapper around the Qdrant client that unifies multiple vector modalities
(dense, sparse and late interaction) behind a simple API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base import AbstractVectorDB

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)

def _batch_iterate(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield successive batches from an iterable."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class QdrantDB(AbstractVectorDB):
    """Wrapper for Qdrant supporting dense, sparse and late interaction vectors."""

    def __init__(
        self,
        collection: str,
        dim: int,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 60,
        distance: str = "COSINE",
        batch_size: int = 8,
        prefer_grpc: bool = False,
    ) -> None:
        if QdrantClient is None:
            raise ImportError("qdrant-client is not installed. Please install it with 'pip install qdrant-client'")
        
        self.collection = collection
        self.dim = dim
        self.distance = getattr(models.Distance, distance)
        self.batch_size = batch_size
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc)

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None) -> None:
        """Upsert points into the collection in batches."""
        qdrant_points = []
        for point in points:
            payload = point.get("payload", {})
            if namespace:
                payload["namespace"] = namespace
            if embed_model:
                payload["embed_model"] = embed_model
            qdrant_points.append(
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=payload
                )
            )

        for batch in _batch_iterate(qdrant_points, self.batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=True,
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Perform a similarity search on the collection."""
        must_conditions = []
        if namespace:
            must_conditions.append(models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace)))
        
        if filter:
            for key, value in filter.items():
                must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{**hit.payload, "score": hit.score} for hit in results]

    def get_points(self, namespace: str, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        if not namespace:
            return []
            
        must_conditions = [models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
        if filter:
            for key, value in filter.items():
                must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        
        query_filter = models.Filter(must=must_conditions)

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        return [p.payload for p in points if p.payload is not None]

    def get_embedding_model(self, namespace: str) -> Optional[str]:
        """Get the name of the embedding model used for a given namespace."""
        if not namespace:
            return None
        
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if points and points[0].payload:
            return points[0].payload.get("embed_model")
        
        return None
