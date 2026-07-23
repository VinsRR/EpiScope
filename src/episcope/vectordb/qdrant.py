from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import AbstractVectorDB

QdrantClient: Any
models: Any
ResponseHandlingException: Any
UnexpectedResponse: Any
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
except ImportError:
    QdrantClient = None
    models = None
    ResponseHandlingException = None
    UnexpectedResponse = None

logger = logging.getLogger(__name__)


def _batch_iterate(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _is_connection_error(exc: Exception) -> bool:
    if ResponseHandlingException is not None and isinstance(
        exc, ResponseHandlingException
    ):
        return True
    return False


def _is_missing_collection_error(exc: Exception) -> bool:
    if UnexpectedResponse is not None and isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    return False


# Set this env var to a truthy value to bypass the client/server version guard
# (advanced use only — running mismatched versions can fail unpredictably).
_SKIP_VERSION_CHECK_ENV = "EPISCOPE_QDRANT_SKIP_VERSION_CHECK"


class QdrantVersionMismatchError(RuntimeError):
    """Raised when the qdrant-client minor differs from the Qdrant server minor.

    Qdrant only guarantees client/server compatibility within the same minor
    version, so a mismatch is treated as a hard error rather than a silent
    source of intermittent failures.
    """

    def __init__(self, client_version: str, server_version: str) -> None:
        server_mm = _parse_major_minor(server_version)
        pin = f"{server_mm[0]}.{server_mm[1]}.0" if server_mm else server_version
        super().__init__(
            f"qdrant-client {client_version} is incompatible with Qdrant server "
            f"{server_version}: the client and server must share the same minor "
            f"version (X.Y.*). Install a matching client, e.g.:\n"
            f"    pip install 'qdrant-client~={pin}'\n"
            f"or align the server image (QDRANT_VERSION in docker-compose.yml). "
            f"Set {_SKIP_VERSION_CHECK_ENV}=1 to bypass this check."
        )


def _parse_major_minor(version: Any) -> Optional[Tuple[int, int]]:
    """Extract ``(major, minor)`` from a version string like ``v1.13.5``."""
    match = re.match(r"v?(\d+)\.(\d+)", str(version or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _fetch_server_version(url: str, api_key: Optional[str], timeout: int) -> Optional[str]:
    """Return the Qdrant server version from its REST root, or ``None``.

    Uses the stdlib so the check adds no dependency to the ``server`` extra.
    """
    root = url.rstrip("/") + "/"
    headers = {"api-key": api_key} if api_key else {}
    request = urllib.request.Request(root, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured URL
        payload = json.loads(response.read().decode("utf-8"))
    version = payload.get("version")
    return str(version) if version else None


def _check_qdrant_version_compatibility(
    url: str, api_key: Optional[str], timeout: int
) -> None:
    """Hard-error on a client/server minor-version mismatch.

    Silently skips when the check is disabled, the versions cannot be
    determined, or the server is unreachable (a real connectivity problem
    surfaces on the first query with a clearer message).
    """
    if os.environ.get(_SKIP_VERSION_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return

    try:
        client_version = importlib.metadata.version("qdrant-client")
    except importlib.metadata.PackageNotFoundError:
        return

    try:
        server_version = _fetch_server_version(url, api_key, timeout)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read Qdrant server version at %s for the compatibility "
            "check (%s); skipping. A real connection problem will surface on the "
            "first query.",
            url,
            exc,
        )
        return

    client_mm = _parse_major_minor(client_version)
    server_mm = _parse_major_minor(server_version)
    if client_mm is None or server_mm is None or client_mm == server_mm:
        return

    raise QdrantVersionMismatchError(client_version, server_version or "unknown")


class QdrantDB(AbstractVectorDB):
    def __init__(
        self,
        collection: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 60,
        batch_size: int = 8,
        prefer_grpc: bool = False,
        *,
        use_dense: bool = True,
        use_sparse: bool = False,
        use_late: bool = False,
        dense_dim: Optional[int] = None,
        dense_distance: str = "COSINE",
        late_dim: Optional[int] = None,
        late_distance: str = "COSINE",
        late_as_reranker: bool = True,
    ) -> None:
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client is not installed. Please install it with 'pip install qdrant-client'"
            )

        if not any([use_dense, use_sparse, use_late]):
            raise ValueError(
                "At least one modality must be enabled: dense, sparse, or late."
            )

        self.collection = collection
        self.batch_size = batch_size
        self.client = QdrantClient(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc
        )
        _check_qdrant_version_compatibility(url, api_key, timeout)

        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_late = use_late

        self.dense_vector_name = "dense"
        self.sparse_vector_name = "sparse"
        self.late_vector_name = "late"

        self.dense_distance = getattr(models.Distance, dense_distance.upper())
        self.late_distance = getattr(models.Distance, late_distance.upper())

        self.dense_dim = dense_dim
        self.late_dim = late_dim
        self._payload_keys: Optional[set[str]] = None

        collection_info = None
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection
            )
        except Exception as exc:
            if _is_connection_error(exc):
                raise ValueError(
                    f"Could not connect to Qdrant at {url} while opening collection "
                    f"{self.collection!r}. Make sure the Qdrant service is running and the "
                    "URL is correct."
                ) from exc
            if not _is_missing_collection_error(exc):
                raise

        if collection_info is not None:
            # Existing collection: inspect capabilities
            vectors_config = collection_info.config.params.vectors
            sparse_vectors_config = getattr(
                collection_info.config.params, "sparse_vectors", None
            )

            if not isinstance(vectors_config, dict):
                raise ValueError(
                    f"Collection '{self.collection}' is not in named-vector mode. "
                    "Please migrate or recreate it."
                )

            self.use_dense = self.dense_vector_name in vectors_config
            self.use_late = self.late_vector_name in vectors_config

            if self.use_dense:
                self.dense_dim = vectors_config[self.dense_vector_name].size
                self.dense_distance = vectors_config[self.dense_vector_name].distance

            if self.use_late:
                self.late_dim = vectors_config[self.late_vector_name].size
                self.late_distance = vectors_config[self.late_vector_name].distance

            self.use_sparse = bool(
                sparse_vectors_config
                and self.sparse_vector_name in sparse_vectors_config
            )
        else:
            vectors_config: Dict[str, Any] = {}
            sparse_vectors_config: Dict[str, Any] = {}

            if use_dense:
                if dense_dim is None:
                    raise ValueError(
                        f"Qdrant collection {self.collection!r} was not found at {url}, "
                        "and dense_dim was not provided to create it."
                    )
                vectors_config[self.dense_vector_name] = models.VectorParams(
                    size=dense_dim,
                    distance=self.dense_distance,
                )

            if use_late:
                if late_dim is None:
                    raise ValueError(
                        "late_dim must be provided when creating a collection with late-interaction vectors."
                    )
                late_kwargs: Dict[str, Any] = {
                    "size": late_dim,
                    "distance": self.late_distance,
                    "multivector_config": models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                }
                if late_as_reranker:
                    late_kwargs["hnsw_config"] = models.HnswConfigDiff(m=0)
                vectors_config[self.late_vector_name] = models.VectorParams(
                    **late_kwargs
                )

            if use_sparse:
                sparse_vectors_config[self.sparse_vector_name] = (
                    models.SparseVectorParams()
                )

            logger.info(
                "Creating collection '%s' with modalities: dense=%s sparse=%s late=%s",
                self.collection,
                use_dense,
                use_sparse,
                use_late,
            )

            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=vectors_config or None,
                sparse_vectors_config=sparse_vectors_config or None,
            )

    def has_dense(self) -> bool:
        return self.use_dense

    def has_sparse(self) -> bool:
        return self.use_sparse

    def has_late(self) -> bool:
        return self.use_late

    def capabilities(self) -> Dict[str, bool]:
        return {
            "dense": self.use_dense,
            "sparse": self.use_sparse,
            "late": self.use_late,
        }

    def upsert(
        self,
        points: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
        embed_models: Optional[Dict[str, str]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if chunking_config:
            db_config = self.get_chunking_config()
            if db_config and db_config != chunking_config:
                raise ValueError(
                    f"Inconsistent chunking config. DB uses '{db_config}', but upsert was called with '{chunking_config}'."
                )

        qdrant_points = []
        for point in points:
            payload = dict(point.get("payload", {}))

            if namespace:
                payload["paper_id"] = namespace
            if embed_models:
                payload["embed_models"] = embed_models
            if chunking_config:
                payload["chunking_config"] = chunking_config

            if self._payload_keys is not None:
                self._payload_keys.update(payload.keys())

            vectors = point.get("vectors", {})
            sparse_vectors = point.get("sparse_vectors", {})

            point_struct_kwargs = {
                "id": point["id"],
                "payload": payload,
            }

            if vectors:
                point_struct_kwargs["vector"] = vectors

            # Depending on qdrant-client version, sparse vectors may be passed as `sparse_vector`
            # or embedded inside the query/upsert structures differently.
            # if sparse_vectors:
            #     point_struct_kwargs["sparse_vector"] = {
            #         name: models.SparseVector(
            #             indices=sv["indices"],
            #             values=sv["values"],
            #         )
            #         for name, sv in sparse_vectors.items()
            #     }
            # Merge sparse vectors into `vector`
            if sparse_vectors:
                if "vector" not in point_struct_kwargs:
                    point_struct_kwargs["vector"] = {}

                for name, sv in sparse_vectors.items():
                    point_struct_kwargs["vector"][name] = models.SparseVector(
                        indices=sv["indices"],
                        values=sv["values"],
                    )

            qdrant_points.append(models.PointStruct(**point_struct_kwargs))

        for batch in _batch_iterate(qdrant_points, self.batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=False,
            )

    def _build_filter(
        self,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        ids: Optional[List[Any]] = None,
    ):
        must_conditions = []

        if namespace:
            must_conditions.append(
                models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=namespace)
                )
            )

        if filter:
            for key, value in filter.items():
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )

        if ids:
            must_conditions.append(models.HasIdCondition(has_id=ids))

        return models.Filter(must=must_conditions) if must_conditions else None

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        if not self.use_dense:
            raise ValueError(
                "Dense search requested, but this collection has no dense vectors."
            )

        query_filter = self._build_filter(namespace=namespace, filter=filter)

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.dense_vector_name,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        points = getattr(results, "points", results)
        return [
            {**(hit.payload or {}), "id": hit.id, "score": hit.score} for hit in points
        ]

    def search_sparse(
        self,
        query_sparse: Dict[str, List[float]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        if not self.use_sparse:
            raise ValueError(
                "Sparse search requested, but this collection has no sparse vectors."
            )

        query_filter = self._build_filter(namespace=namespace, filter=filter)

        results = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(
                indices=query_sparse["indices"],
                values=query_sparse["values"],
            ),
            using=self.sparse_vector_name,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        points = getattr(results, "points", results)
        return [
            {**(hit.payload or {}), "id": hit.id, "score": hit.score} for hit in points
        ]

    def search_hybrid(
        self, dense_query, sparse_query, top_k, prefetch_k, namespace=None, filter=None
    ):
        if not self.has_dense() or not self.has_sparse():
            raise ValueError(
                "Hybrid search requires both dense and sparse vectors in the collection."
            )

        query_filter = self._build_filter(namespace=namespace, filter=filter)

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=self.dense_vector_name,
                    limit=prefetch_k,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using=self.sparse_vector_name,
                    limit=prefetch_k,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        points = getattr(results, "points", results)
        return [
            {**(hit.payload or {}), "id": hit.id, "score": hit.score} for hit in points
        ]

    # def search_hybrid(
    #     self,
    #     dense_query: List[float],
    #     sparse_query: Dict[str, List[float]],
    #     top_k: int = 5,
    #     prefetch_k: int = 50,
    #     namespace: Optional[str] = None,
    #     filter: Optional[Dict[str, Any]] = None,
    # ) -> Sequence[Dict[str, Any]]:
    #     if not (self.use_dense and self.use_sparse):
    #         raise ValueError("Hybrid search requires both dense and sparse vectors in the collection.")

    #     query_filter = self._build_filter(namespace=namespace, filter=filter)

    #     results = self.client.query_points(
    #         collection_name=self.collection,
    #         prefetch=[
    #             models.Prefetch(
    #                 query=models.NamedVector(name=self.dense_vector_name, vector=dense_query),
    #                 using=self.dense_vector_name,
    #                 limit=prefetch_k,
    #                 filter=query_filter,
    #             ),
    #             models.Prefetch(
    #                 query=models.SparseVector(
    #                     indices=sparse_query["indices"],
    #                     values=sparse_query["values"],
    #                 ),
    #                 using=self.sparse_vector_name,
    #                 limit=prefetch_k,
    #                 filter=query_filter,
    #             ),
    #         ],
    #         query=models.FusionQuery(fusion=models.Fusion.RRF),
    #         limit=top_k,
    #         with_payload=True,
    #     )
    #     points = getattr(results, "points", results)
    #     return [{**(hit.payload or {}), "id": hit.id, "score": hit.score} for hit in points]

    def rerank_late(
        self,
        query_late: List[List[float]],
        candidate_ids: List[Any],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        if not self.use_late:
            raise ValueError(
                "Late-interaction reranking requested, but this collection has no late vectors."
            )

        query_filter = self._build_filter(
            namespace=namespace, filter=filter, ids=candidate_ids
        )

        # Depending on client version, `query_points` may accept multivector query directly through `using=...`.
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_late,
            using=self.late_vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(results, "points", results)
        return [
            {**(hit.payload or {}), "id": hit.id, "score": hit.score} for hit in points
        ]

    # Compatibility alias: old code keeps working as dense retrieval.
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        distance: Optional[str] = None,
    ) -> Sequence[Dict[str, Any]]:
        return self.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )

    def get_points(
        self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None
    ) -> Sequence[Dict[str, Any]]:
        query_filter = self._build_filter(namespace=namespace, filter=filter)
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload for p in points if p.payload is not None]

    def delete(self, namespace: str) -> int:
        query_filter = self._build_filter(namespace=namespace)
        if query_filter is None:
            return 0
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )
        return len(points)

    # def get_embedding_model(self) -> Optional[str]:
    #     """Get the name of the embedding model used for the database."""
    #     points, _ = self.client.scroll(
    #         collection_name=self.collection,
    #         limit=1,
    #         with_payload=True,
    #         with_vectors=False
    #     )

    #     if points and points[0].payload:
    #         print(points[0].payload)
    #         return points[0].payload.get("embed_model")

    #     return None

    def get_embedding_model(self) -> Dict[str, str]:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if points and points[0].payload:
            if "embed_models" in points[0].payload:
                return points[0].payload.get("embed_models", {})
            elif "embed_model" in points[0].payload:
                return points[0].payload.get("embed_model")
            else:
                assert False, (
                    "No 'embed_model' or 'embed_models' key found in payload. Payload keys: "
                    + ", ".join(points[0].payload.keys())
                )
        return {}

    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if points and points[0].payload:
            return points[0].payload.get("chunking_config")
        return None

    def get_payload_keys(self) -> set[str]:
        if self._payload_keys is not None:
            return self._payload_keys

        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        keys = set()
        for point in points:
            if point.payload:
                keys.update(point.payload.keys())

        self._payload_keys = keys
        return self._payload_keys
