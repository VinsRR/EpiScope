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
        dim: Optional[int] = None,
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
        self.default_distance = distance.upper()
        self.batch_size = batch_size
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc)
        self._payload_keys: Optional[set[str]] = None
        
        self.vector_names = {
            "COSINE": "vector_cosine",
            "EUCLID": "vector_euclid",
            "DOT": "vector_dot",
            "MANHATTAN": "vector_manhattan",
        }
        self._single_vector_mode = False
        self._single_vector_distance: Optional[models.Distance] = None

        try:
            collection_info = self.client.get_collection(collection_name=self.collection)
            vectors_config = collection_info.config.params.vectors
            
            if isinstance(vectors_config, models.VectorParams):
                self._single_vector_mode = True
                self._single_vector_distance = vectors_config.distance
                collection_dim = vectors_config.size
                logger.warning(
                    f"Collection '{self.collection}' uses a single vector configuration. "
                    f"Only {self._single_vector_distance.name} distance is supported for searching."
                )
            else:
                existing_vector_name = next((name for name in self.vector_names.values() if name in vectors_config), None)
                if not existing_vector_name:
                    raise ValueError(f"Collection '{self.collection}' has no recognized named vectors.")
                collection_dim = vectors_config[existing_vector_name].size

            if dim is not None and dim != collection_dim:
                logger.warning(
                    f"Dimension mismatch for collection '{self.collection}'. "
                    f"Provided: {dim}, Existing: {collection_dim}. "
                    f"Using existing dimension: {collection_dim}."
                )
            self.dim = collection_dim
        except Exception:
            if dim is None:
                raise ValueError(f"Dimension 'dim' must be provided to create collection '{self.collection}'.")
            
            logger.info(f"Collection '{self.collection}' not found. Creating a new one with dimension {dim} and multiple distance metrics.")
            self.dim = dim
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config={
                    self.vector_names["COSINE"]: models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                    self.vector_names["EUCLID"]: models.VectorParams(size=self.dim, distance=models.Distance.EUCLID),
                    self.vector_names["DOT"]: models.VectorParams(size=self.dim, distance=models.Distance.DOT),
                    self.vector_names["MANHATTAN"]: models.VectorParams(size=self.dim, distance=models.Distance.MANHATTAN),
                },
            )

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None, chunking_config: Optional[Dict[str, Any]] = None) -> None:
        """Upsert points into the collection in batches."""
        if chunking_config:
            db_config = self.get_chunking_config()
            if db_config and db_config != chunking_config:
                raise ValueError(f"Inconsistent chunking config. DB uses '{db_config}', but upsert was called with '{chunking_config}'.")

        qdrant_points = []
        for point in points:
            payload = point.get("payload", {})
            if namespace:
                payload["paper_id"] = namespace
            if embed_model:
                payload["embed_model"] = embed_model
            if chunking_config:
                payload["chunking_config"] = chunking_config
            
            if self._payload_keys is not None:
                self._payload_keys.update(payload.keys())

            vector_data = point["vector"]
            vector_to_upsert = (
                vector_data
                if self._single_vector_mode
                else {name: vector_data for name in self.vector_names.values()}
            )

            qdrant_points.append(
                models.PointStruct(
                    id=point["id"],
                    vector=vector_to_upsert,
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
        distance: Optional[str] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Perform a similarity search on the collection."""
        must_conditions = []
        if namespace:
            must_conditions.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=namespace)))
        
        if filter:
            for key, value in filter.items():
                must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        search_distance = (distance or self.default_distance).upper()

        if self._single_vector_mode:
            if self._single_vector_distance and search_distance != self._single_vector_distance.name:
                logger.warning(
                    f"Searching with distance {search_distance} but collection only supports {self._single_vector_distance.name}. "
                    f"Using {self._single_vector_distance.name} for search."
                )
            query_vector_to_search = query_vector
        else:
            if search_distance not in self.vector_names:
                raise ValueError(f"Unsupported distance metric: {search_distance}. Supported are: {list(self.vector_names.keys())}")
            vector_name = self.vector_names[search_distance]
            query_vector_to_search = models.NamedVector(name=vector_name, vector=query_vector)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector_to_search,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{**hit.payload, "score": hit.score} for hit in results]

    def get_points(self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        must_conditions = []
        if namespace:
            must_conditions.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=namespace)))
        if filter:
            for key, value in filter.items():
                must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        return [p.payload for p in points if p.payload is not None]

    def get_embedding_model(self) -> Optional[str]:
        """Get the name of the embedding model used for the database."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if points and points[0].payload:
            return points[0].payload.get("embed_model")
        
        return None

    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        """Get the chunking configuration used for the database."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if points and points[0].payload:
            return points[0].payload.get("chunking_config")
        
        return None

    def get_payload_keys(self) -> set[str]:
        """Get the set of all available payload keys by sampling a few records."""
        if self._payload_keys is not None:
            return self._payload_keys

        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        keys = set()
        for point in points:
            if point.payload:
                keys.update(point.payload.keys())
        
        self._payload_keys = keys
        return self._payload_keys
