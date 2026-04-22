from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from episcope.clients import LLMClient
from episcope.rag.postprocessing.reranker import Reranker
from episcope.rag.retrieval.base import BaseRetriever
from episcope.rag.retrieval.candidates import (
    HybridCandidateRetriever,
    SemanticCandidateRetriever,
    SparseCandidateRetriever,
)
from episcope.rag.retrieval.components import CandidateRetriever, FusionStrategy, QueryTransformer
from episcope.rag.retrieval.fusion import RRFFusion
from episcope.rag.retrieval.rerankers import LateInteractionReranker
from episcope.rag.retrieval.transforms import HyDEQueryTransformer
from episcope.schemas import SearchResult
from episcope.vectordb.base import AbstractVectorDB

logger = logging.getLogger(__name__)


class Retriever(BaseRetriever):
    """Workflow-facing retriever orchestrating reusable retrieval stages."""

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        prefetch_k: int = 50,
        rerank_top_k: int = 20,
        use_rerank: bool = True,
        cross_encoder: Optional[Reranker] = None,
        rrf_k: int = 60,
        *,
        query_transformers: Optional[Sequence[QueryTransformer]] = None,
        candidate_retrievers: Optional[Sequence[CandidateRetriever]] = None,
        fusion: Optional[FusionStrategy] = None,
        late_reranker: Optional[LateInteractionReranker] = None,
        hyde: Optional[bool] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(vectordb)
        self.prefetch_k = prefetch_k
        self.rerank_top_k = rerank_top_k
        self.use_rerank = use_rerank
        self.cross_encoder = cross_encoder
        self.rrf_k = rrf_k

        self.query_transformers = list(query_transformers or [])
        if hyde:
            self.query_transformers.append(HyDEQueryTransformer(hyde=True, llm_client=llm_client))

        self.candidate_retrievers = list(candidate_retrievers or self._build_default_candidate_retrievers())
        self.fusion = fusion or (RRFFusion(k=rrf_k) if len(self.candidate_retrievers) > 1 else None)
        self.late_reranker = late_reranker or self._build_default_late_reranker()

    @property
    def uses_cross_encoder(self) -> bool:
        return self.cross_encoder is not None

    def _build_default_candidate_retrievers(self) -> List[CandidateRetriever]:
        caps = self.vectordb.capabilities()
        retrievers: List[CandidateRetriever] = []

        if caps.get("dense") and caps.get("sparse"):
            retrievers.append(HybridCandidateRetriever(self.vectordb, prefetch_k=self.prefetch_k))
        elif caps.get("dense"):
            retrievers.append(SemanticCandidateRetriever(self.vectordb))
        elif caps.get("sparse"):
            retrievers.append(SparseCandidateRetriever(self.vectordb))
        if not retrievers:
            raise ValueError("No usable retrieval modality is available in the collection.")

        return retrievers

    def _build_default_late_reranker(self) -> Optional[LateInteractionReranker]:
        caps = self.vectordb.capabilities()
        if not self.use_rerank or self.cross_encoder is not None or not caps.get("late", False):
            return None
        return LateInteractionReranker(self.vectordb)

    def _transform_query(self, query: str) -> str:
        transformed = query
        for transformer in self.query_transformers:
            transformed = transformer.transform(transformed)
        return transformed

    def _retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str],
        filter: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], str]:
        if not self.candidate_retrievers:
            raise ValueError("Retriever requires at least one candidate retriever.")

        if len(self.candidate_retrievers) == 1:
            retriever = self.candidate_retrievers[0]
            candidates = retriever.retrieve_candidates(
                query,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )
            source = getattr(retriever, "source", self.default_source)
            return list(candidates), source

        if self.fusion is None:
            raise ValueError("Multiple candidate retrievers require a fusion strategy.")

        results_lists = [
            retriever.retrieve_candidates(
                query,
                top_k=self.prefetch_k,
                namespace=namespace,
                filter=filter,
            )
            for retriever in self.candidate_retrievers
        ]
        return self.fusion.fuse(results_lists, top_k=top_k), "hybrid"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        namespace, final_filter = self._prepare_filter(filter)

        use_cross = self.use_rerank and self.cross_encoder is not None
        use_late = self.use_rerank and not use_cross and self.late_reranker is not None
        initial_top_k = self.rerank_top_k if (use_cross or use_late) else top_k
        retrieval_query = self._transform_query(query)

        candidates, source = self._retrieve_candidates(
            retrieval_query,
            top_k=initial_top_k,
            namespace=namespace,
            filter=final_filter if final_filter else None,
        )

        if use_late and self.late_reranker is not None:
            candidates = self.late_reranker.rerank_candidates(
                query,
                candidates,
                top_k=top_k,
                namespace=namespace,
                filter=final_filter if final_filter else None,
            )
        else:
            candidates = candidates[:top_k]

        results = self._to_search_results(
            candidates,
            source=source,
            similarity_threshold=similarity_threshold,
        )

        if use_cross and self.cross_encoder is not None:
            return self.cross_encoder.rerank(query, list(results), top_k)

        return results
