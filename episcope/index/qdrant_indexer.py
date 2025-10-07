"""Qdrant‑based indexer implementation.

This module provides an implementation of the :class:`AbstractIndexer`
that uses the Qdrant vector database to store and search embeddings.
It leverages the ``UnifiedQdrantIndex`` wrapper defined in
``episcope.index.unified_db`` to support dense, sparse and late
interaction vectors.  When Qdrant is unavailable, the indexer
automatically falls back to an in‑memory :class:`FaissIndexer` so
that test environments without the Qdrant client can still import
and exercise the code.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Dict, Any, Optional, List

from ..core.interfaces import AbstractIndexer
from ..retrieve.embeddings import SimplifiedEmbedder
from .faiss_indexer import FaissIndexer

try:
    from .unified_db import UnifiedQdrantIndex  # type: ignore
    _HAS_QDRANT = True
except Exception:
    # Provide dummy sentinel to allow import
    UnifiedQdrantIndex = None  # type: ignore
    _HAS_QDRANT = False

logger = logging.getLogger(__name__)


class QdrantIndexer(AbstractIndexer):
    """Indexer backed by a Qdrant vector database.

    This indexer stores vectors in Qdrant when the client library is
    available.  Otherwise it transparently falls back to a
    :class:`FaissIndexer`.  The indexer uses a
    :class:`SimplifiedEmbedder` to compute embeddings and stores
    payloads as Qdrant point payloads.
    """

    def __init__(
        self,
        collection: str = "episcope",
        embed_model: str = "distilbert-base-uncased",
        batch_size: int = 8,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        qdrant_timeout: int = 60,
    ) -> None:
        self.embedder = SimplifiedEmbedder(embed_model=embed_model, batch_size=batch_size)
        if _HAS_QDRANT:
            # Determine approximate dimension from embedder
            dim = self.embedder.dim
            self._index = UnifiedQdrantIndex(
                collection=collection,
                dim=dim,
                dim_late_interaction=None,
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=qdrant_timeout,
            )
            # Attempt to create the collection with default config
            try:
                self._index.create_collection(multivector=False, sparse=False, late_interaction=False)
                logger.info(f"Created Qdrant collection '{collection}'")
            except Exception as e:
                logger.warning(f"Could not create Qdrant collection '{collection}': {e}")
        else:
            # Fallback to in‑memory FaissIndexer
            self._index = FaissIndexer(embed_model=embed_model, batch_size=batch_size)  # type: ignore
        self._is_qdrant = _HAS_QDRANT

    def index_documents(self, docs: Iterable[Dict[str, Any]], *, namespace: Optional[str] = None) -> None:
        docs_list = list(docs)
        if not docs_list:
            return
        if self._is_qdrant:
            # Compute embeddings
            texts = [d.get("content", "") for d in docs_list]
            dense_embs = self.embedder.embed_texts(texts)
            # Normalize embeddings for cosine similarity; Qdrant expects raw vectors for dot product
            # But unify by normalizing to ensure comparability with fallback
            import numpy as np
            dense_embs = np.array(dense_embs, dtype="float32")
            norms = np.linalg.norm(dense_embs, axis=1, keepdims=True) + 1e-9
            dense_embs = dense_embs / norms
            points = []
            for vec, doc in zip(dense_embs, docs_list):
                payload = {k: v for k, v in doc.items() if k != "content"}
                points.append({"vector": vec.tolist(), "payload": payload})
            try:
                self._index.upsert(points)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Failed to upsert points into Qdrant: {e}; falling back to FAISS")
                # Fallback: index into local FAISS indexer
                self._index = FaissIndexer(embed_model=self.embedder._model_name, batch_size=self.embedder._batch_size)  # type: ignore
                self._is_qdrant = False
                self.index_documents(docs_list, namespace=namespace)
        else:
            # Delegate to fallback indexer
            self._index.index_documents(docs_list, namespace=namespace)

    def search(self, query: str, *, top_k: int = 5, namespace: Optional[str] = None) -> Sequence[Dict[str, Any]]:
        if self._is_qdrant:
            # Compute embedding for query
            import numpy as np
            q_emb = np.array(self.embedder.embed_text(query), dtype="float32")
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            try:
                # Qdrant expects list of dicts with vectors and filter, but unified wrapper hides details
                search_res = self._index.search(q_emb.tolist(), top_k)  # type: ignore[no-untyped-call]
                # Each result is expected to contain payload and score keys
                return search_res
            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}; falling back to FAISS")
                # convert to fallback indexer
                self._index = FaissIndexer(embed_model=self.embedder._model_name, batch_size=self.embedder._batch_size)  # type: ignore
                self._is_qdrant = False
                # We cannot recover embeddings for previous docs; warn
                return []
        else:
            return self._index.search(query, top_k=top_k, namespace=namespace)  # type: ignore[attr-defined]
