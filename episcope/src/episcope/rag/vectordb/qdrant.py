"""
index/unified_db.py

Wrapper around the Qdrant client that unifies multiple vector modalities
(dense, sparse and late interaction) behind a simple API.  This class is
based on the original ``QdrantUnifiedDB`` from the ``rag_tool`` project
but refactored to improve readability and testability.

The public methods include:
  - ``create_collection``: create a new collection with optional sparse
    and late‑interaction vectors.
  - ``upsert``: batch upsert points into the collection.
  - ``search``: perform a similarity search over the collection.

See https://qdrant.tech/ for details on the vector database.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional



# Attempt to import Qdrant; provide fallbacks if unavailable.
try:
    from qdrant_client import QdrantClient  # type: ignore[assignment]
    from qdrant_client.http import models  # type: ignore[assignment]
except Exception:
    # Provide minimal stubs so that the module can be imported in test
    import types

    QdrantClient = None  # type: ignore[assignment]

    class _DummyDistance:
        COSINE = "COSINE"

    class _DummyVectorParams:
        def __init__(self, size: int, distance: str, on_disk: bool = True, multivector_config: Any = None):
            self.size = size
            self.distance = distance
            self.on_disk = on_disk
            self.multivector_config = multivector_config

    class _DummyMultiVectorComparator:
        MAX_SIM = "MAX_SIM"

    class _DummyMultiVectorConfig:
        def __init__(self, comparator: str):
            self.comparator = comparator

    class _DummySparseVectorParams:
        def __init__(self, modifier: Any):
            self.modifier = modifier

    class _DummyModifier:
        IDF = "IDF"

    class _DummyOptimizersConfigDiff:
        pass

    class _DummyBinaryQuantization:
        pass

    class _DummyPointStruct:
        pass

    class _DummySearchParams:
        pass

    class _DummyFilter:
        pass

    models = types.SimpleNamespace(
        Distance=_DummyDistance,
        VectorParams=_DummyVectorParams,
        MultiVectorComparator=_DummyMultiVectorComparator,
        MultiVectorConfig=_DummyMultiVectorConfig,
        SparseVectorParams=_DummySparseVectorParams,
        Modifier=_DummyModifier,
        OptimizersConfigDiff=_DummyOptimizersConfigDiff,
        BinaryQuantization=_DummyBinaryQuantization,
        PointStruct=_DummyPointStruct,
        SearchParams=_DummySearchParams,
        Filter=_DummyFilter,
    )


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


from .base import AbstractVectorDB

class QdrantDB(AbstractVectorDB):
    """Wrapper for Qdrant supporting dense, sparse and late interaction vectors."""

    def __init__(
        self,
        collection: str,
        dim: int,
        dim_late_interaction: Optional[int] = None,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 60,
        distance: str = "COSINE",
        batch_size: int = 8,
        prefer_grpc: bool = False,
    ) -> None:
        self.collection = collection
        self.dim = dim
        self.dim_late_interaction = dim_late_interaction
        self.distance = getattr(models.Distance, distance)
        self.batch_size = batch_size
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )
        # self.create_collection()

    def create_collection(
        self,
        multivector: bool = False,
        on_disk: bool = True,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        quantization_config: Optional[models.BinaryQuantization] = None,
        sparse: bool = False,
        late_interaction: bool = False,
    ) -> None:
        """Create the collection if it does not already exist."""
        if self.client.collection_exists(self.collection):
            return

        # Primary dense vector configuration
        dense_params = models.VectorParams(
            size=self.dim,
            distance=self.distance,
            on_disk=on_disk,
        )

        if multivector:
            dense_params.multivector_config = models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )

        # Build the vector configuration
        if late_interaction:
            if not self.dim_late_interaction:
                raise ValueError(
                    "dim_late_interaction must be provided when late_interaction is True"
                )
            vectors_config: Dict[str, models.VectorParams] = {
                "dense": dense_params,
                "late_interaction": models.VectorParams(
                    size=self.dim_late_interaction,
                    distance=self.distance,
                    on_disk=on_disk,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            }
        else:
            vectors_config = dense_params

        sparse_config: Optional[Dict[str, models.SparseVectorParams]] = None
        if sparse:
            sparse_config = {
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            optimizers_config=optimizers_config,
            quantization_config=quantization_config,
            on_disk_payload=on_disk,
        )

    def upsert(self, points: Iterable[models.PointStruct], namespace: Optional[str] = None) -> None:
        """Upsert points into the collection in batches."""
        # The namespace is handled at the payload level in Qdrant
        for batch in _batch_iterate(points, self.batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=list(batch),
                wait=True,
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Perform a similarity search on the collection."""
        query_filter = None
        if namespace:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
            )
        
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            {**hit.payload, "score": hit.score} for hit in results
        ]

