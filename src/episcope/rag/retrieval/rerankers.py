from __future__ import annotations

from typing import Any, Dict, List, Optional

from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.retrieval.candidates import _embedding_models
from episcope.rag.retrieval.components import CandidateReranker
from episcope.vectordb.base import AbstractVectorDB


class LateInteractionReranker(CandidateReranker):
    """Rerank candidates using the vector DB late-interaction index."""

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        *,
        late_embedder: Optional[Any] = None,
    ) -> None:
        self.vectordb = vectordb

        model_name = _embedding_models(vectordb).get("late")
        if late_embedder is not None:
            self.late_embedder = late_embedder
        else:
            if not model_name:
                raise ValueError(
                    "VectorDB does not have a late interaction model configured."
                )
            self.late_embedder = EmbedderFactory.get_late_embedder(model_name)

    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_late = self.late_embedder.embed_query(query)
        candidate_ids = [item["id"] for item in candidates if "id" in item]
        return list(
            self.vectordb.rerank_late(
                query_late=query_late,
                candidate_ids=candidate_ids,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )
        )
