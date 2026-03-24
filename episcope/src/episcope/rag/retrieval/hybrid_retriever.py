from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Sequence

from episcope.rag.embeddings.base import Embedder
from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.postprocessing.reranker import Reranker
from episcope.schemas.results import SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever(AbstractRetriever):
    def __init__(
        self,
        vdb,
        prefetch_k: int = 50,
        rerank_top_k: int = 20,
        use_rerank: bool = True,
        cross_encoder: Optional[Reranker] = None,
        rrf_k: int = 60,
    ) -> None:
        self.vdb = vdb
        self.prefetch_k = prefetch_k
        self.rerank_top_k = rerank_top_k
        self.use_rerank = use_rerank
        self.cross_encoder = cross_encoder
        self.rrf_k = rrf_k

        self._init_models()

    def _init_models(self) -> None:
        models = {}
        if hasattr(self.vdb, "get_embedding_model"):
            models = self.vdb.get_embedding_model()

        # Dense
        self.dense_embedder = None
        if models.get("dense"):
            model_name = models["dense"]
            if "embedding-001" in model_name:
                model_name = "gemini-embedding-001"
            logger.info(f"Auto-loading dense embedder: {model_name}")
            self.dense_embedder = EmbedderFactory.get_embedder(model_name)

        # Sparse
        self.sparse_embedder = None
        if models.get("sparse"):
            logger.info(f"Auto-loading sparse embedder: {models['sparse']}")
            self.sparse_embedder = EmbedderFactory.get_sparse_embedder(models["sparse"])

        # Late
        self.late_embedder = None
        if models.get("late"):
            logger.info(f"Auto-loading late embedder: {models['late']}")
            self.late_embedder = EmbedderFactory.get_late_embedder(models["late"])

    @property
    def uses_cross_encoder(self) -> bool:
        return self.cross_encoder is not None

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self, results_lists: List[List[Dict[str, Any]]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fuses multiple lists of results using Reciprocal Rank Fusion (RRF)."""
        fused_scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for results in results_lists:
            for rank, item in enumerate(results):
                doc_id = item.get("id")
                if doc_id is None:
                    continue
                if doc_id not in items:
                    items[doc_id] = item
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            item = dict(items[doc_id])
            item["rrf_score"] = rrf_score
            fused_results.append(item)

        return fused_results

    def _get_dense_candidates(
        self, query: str, top_k: int, namespace: Optional[str], filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.dense_embedder:
            raise ValueError("Dense retrieval requested, but no dense embedder was provided.")
        dense_query = self.dense_embedder.embed_text(query)
        return list(self.vdb.search_dense(
            query_vector=dense_query,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        ))

    def _get_sparse_candidates(
        self, query: str, top_k: int, namespace: Optional[str], filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.sparse_embedder:
            raise ValueError("Sparse retrieval requested, but no sparse embedder was provided.")
        sparse_query = self.sparse_embedder.embed_text(query)
        return list(self.vdb.search_sparse(
            query_sparse=sparse_query,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        ))

    def _get_hybrid_candidates(
        self, query: str, top_k: int, namespace: Optional[str], filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.dense_embedder or not self.sparse_embedder:
            raise ValueError("Hybrid retrieval requires both dense and sparse embedders.")

        if hasattr(self.vdb, "search_hybrid"):
            dense_query = self.dense_embedder.embed_text(query)
            sparse_query = self.sparse_embedder.embed_text(query)
            return list(self.vdb.search_hybrid(
                dense_query=dense_query,
                sparse_query=sparse_query,
                top_k=top_k,
                prefetch_k=self.prefetch_k,
                namespace=namespace,
                filter=filter,
            ))
        else:
            dense_results = self._get_dense_candidates(query, self.prefetch_k, namespace, filter)
            sparse_results = self._get_sparse_candidates(query, self.prefetch_k, namespace, filter)
            return self._reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)



    def _rerank_cross(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        source: str,
        similarity_threshold: float,
        top_k: int,
    ) -> List[SearchResult]:
        """Convert candidates to SearchResult, apply threshold, then rerank.

        Threshold is applied before the cross-encoder so it never scores
        passages that would be discarded anyway.
        """
        as_results = self._to_search_results(candidates, source, similarity_threshold)
        return self.cross_encoder.rerank(query, list(as_results), top_k)

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

        query_late = self.late_embedder.embed_query(query)
        candidate_ids = [item["id"] for item in candidates]

        return list(self.vdb.rerank_late(
            query_late=query_late,
            candidate_ids=candidate_ids,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        ))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_search_results(
        self,
        candidates: List[Dict[str, Any]],
        source: str,
        similarity_threshold: float,
    ) -> List[SearchResult]:
        results = []
        for chunk in candidates:
            score = chunk.get("score", 0.0)
            if score < similarity_threshold:
                continue
            rank_score = chunk.get("rrf_score", chunk.get("rank_score", score))
            results.append(
                SearchResult(
                    id=str(chunk.get("id", "")),
                    paper_id=chunk.get("paper_id", ""),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=score,
                    rank_score=rank_score,
                    source=source,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        caps = self.vdb.capabilities()

        final_filter = filter.copy() if filter else {}
        namespace = final_filter.pop("paper_id", None) or final_filter.pop("namespace", None)

        use_cross = self.use_rerank and self.cross_encoder is not None
        use_late = self.use_rerank and not use_cross and caps.get("late", False)
        initial_top_k = self.rerank_top_k if (use_cross or use_late) else top_k

        # --- Fetch candidates (always raw dicts from the vdb) -----------
        if caps.get("dense") and caps.get("sparse"):
            logger.info("Using hybrid retrieval (dense + sparse)")
            source = "hybrid"
            candidates = self._get_hybrid_candidates(query, initial_top_k, namespace, final_filter)
        elif caps.get("dense"):
            logger.info("Using dense retrieval")
            source = "dense"
            candidates = self._get_dense_candidates(query, initial_top_k, namespace, final_filter)
        elif caps.get("sparse"):
            logger.info("Using sparse retrieval")
            source = "sparse"
            candidates = self._get_sparse_candidates(query, initial_top_k, namespace, final_filter)
        else:
            raise ValueError("No usable retrieval modality is available in the collection.")

        # --- Rerank and return ------------------------------------------
        if use_cross:
            # _rerank_cross handles conversion and returns List[SearchResult] directly.
            return self._rerank_cross(query, candidates, source, similarity_threshold, top_k)

        if use_late:
            candidates = self._rerank_late(query, candidates, top_k, namespace, final_filter)
        else:
            candidates = candidates[:top_k]

        return self._to_search_results(candidates, source, similarity_threshold)

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        final_filter = filter.copy() if filter else {}
        final_filter["paper_id"] = paper_id
        return self.retrieve(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=final_filter,
        )