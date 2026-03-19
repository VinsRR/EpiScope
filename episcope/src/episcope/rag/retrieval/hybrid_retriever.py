from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from episcope.rag.embeddings.base import Embedder
from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.embeddings.huggingface import HuggingFaceCrossEncoderReranker, HuggingFaceLateEmbedder, HuggingFaceSparseEmbedder

logger = logging.getLogger(__name__)


class HybridRetriever(AbstractRetriever):
    def __init__(
        self,
        vdb,
        dense_embedder: Optional[Embedder] = None,
        sparse_embedder: Optional[HuggingFaceSparseEmbedder] = None,
        late_embedder: Optional[HuggingFaceLateEmbedder] = None,
        cross_encoder_reranker: Optional[HuggingFaceCrossEncoderReranker] = None,
        cross_encoder_model: Optional[str] = None,
        *,
        prefetch_k: int = 50,
        rerank_top_k: int = 20,
        use_rerank: bool = True,
        cross_encoder_batch_size: int = 16,
        cross_encoder_max_length: Optional[int] = None,
        cross_encoder_text_key: str = "text",
    ) -> None:
        self.vdb = vdb

        models = {}
        if hasattr(self.vdb, "get_embedding_model"):
            models = self.vdb.get_embedding_model()

        if dense_embedder is None and models.get("dense"):
            model_name = models["dense"]
            if "embedding-001" in model_name:
                model_name = "gemini-embedding-001"
            logger.info(f"Auto-loading dense embedder: {model_name}")
            self.dense_embedder = EmbedderFactory.get_embedder(model_name)
        else:
            self.dense_embedder = dense_embedder

        if sparse_embedder is None and models.get("sparse"):
            logger.info(f"Auto-loading sparse embedder: {models['sparse']}")
            self.sparse_embedder = EmbedderFactory.get_sparse_embedder(models["sparse"])
        else:
            self.sparse_embedder = sparse_embedder

        if late_embedder is None and models.get("late"):
            logger.info(f"Auto-loading late embedder: {models['late']}")
            self.late_embedder = EmbedderFactory.get_late_embedder(models["late"])
        else:
            self.late_embedder = late_embedder

        if cross_encoder_reranker is not None and cross_encoder_model is not None:
            raise ValueError("Provide either cross_encoder_reranker or cross_encoder_model, not both.")

        if cross_encoder_reranker is not None:
            self.cross_encoder_reranker = cross_encoder_reranker
        else:
            self.cross_encoder_reranker = None

        self.prefetch_k = prefetch_k
        self.rerank_top_k = rerank_top_k
        self.use_rerank = use_rerank
        self.cross_encoder_text_key = cross_encoder_text_key

    @property
    def uses_cross_encoder(self) -> bool:
        return self.cross_encoder_reranker is not None

    def _rerank_cross(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return self.cross_encoder_reranker.rerank(
            query=query,
            candidates=candidates,
            text_key=self.cross_encoder_text_key,
            top_k=top_k,
        )

    def _rerank_late(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.late_embedder is None:
            raise ValueError("Late reranking requested, but no late embedder was provided.")

        query_late = self.late_embedder.embed_text(query)
        candidate_ids = [item["id"] for item in candidates]

        return list(self.vdb.rerank_late(
            query_late=query_late,
            candidate_ids=candidate_ids,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        ))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        caps = self.vdb.capabilities()

        use_cross = self.use_rerank and self.cross_encoder_reranker is not None
        use_late = self.use_rerank and not use_cross and caps.get("late", False)

        initial_top_k = self.rerank_top_k if (use_cross or use_late) else top_k

        if caps["dense"] and caps["sparse"]:
            if self.dense_embedder is None or self.sparse_embedder is None:
                raise ValueError("Hybrid retrieval requires both dense and sparse embedders.")

            dense_query = self.dense_embedder.embed_text(query)
            sparse_query = self.sparse_embedder.embed_text(query)

            fused = list(self.vdb.search_hybrid(
                dense_query=dense_query,
                sparse_query=sparse_query,
                top_k=initial_top_k,
                prefetch_k=self.prefetch_k,
                namespace=namespace,
                filter=filter,
            ))

            if use_cross:
                return self._rerank_cross(query=query, candidates=fused, top_k=top_k)

            if use_late:
                return self._rerank_late(
                    query=query,
                    candidates=fused,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter,
                )

            return fused

        if caps["dense"]:
            if self.dense_embedder is None:
                raise ValueError("Dense retrieval requested, but no dense embedder was provided.")

            dense_query = self.dense_embedder.embed_text(query)
            results = list(self.vdb.search_dense(
                query_vector=dense_query,
                top_k=initial_top_k,
                namespace=namespace,
                filter=filter,
            ))

            if use_cross:
                return self._rerank_cross(query=query, candidates=results, top_k=top_k)

            return results[:top_k]

        if caps["sparse"]:
            if self.sparse_embedder is None:
                raise ValueError("Sparse retrieval requested, but no sparse embedder was provided.")

            sparse_query = self.sparse_embedder.embed_text(query)
            results = list(self.vdb.search_sparse(
                query_sparse=sparse_query,
                top_k=initial_top_k,
                namespace=namespace,
                filter=filter,
            ))

            if use_cross:
                return self._rerank_cross(query=query, candidates=results, top_k=top_k)

            return results[:top_k]

        raise ValueError("No usable retrieval modality is available in the collection.")