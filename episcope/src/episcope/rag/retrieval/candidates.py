from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.retrieval.base import BaseRetriever
from episcope.rag.retrieval.components import CandidateRetriever, FusionStrategy, QueryTransformer
from episcope.rag.retrieval.fusion import RRFFusion
from episcope.schemas import SearchResult
from episcope.vectordb.base import AbstractVectorDB

logger = logging.getLogger(__name__)


def _embedding_models(vectordb: AbstractVectorDB) -> Dict[str, str]:
    models = vectordb.get_embedding_model()
    if isinstance(models, dict):
        return models
    if isinstance(models, str):
        return {"dense": models}
    return {}


def _normalize_dense_model_name(model_name: str) -> str:
    if "embedding-001" in model_name:
        return "gemini-embedding-001"
    return model_name


class SemanticCandidateRetriever(BaseRetriever, CandidateRetriever):
    """Dense-only candidate retriever, also usable as a standalone retriever."""

    default_source = "semantic"
    source = "semantic"

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        *,
        dense_embedder: Optional[Any] = None,
        query_transformers: Optional[Sequence[QueryTransformer]] = None,
    ) -> None:
        super().__init__(vectordb)
        self.query_transformers = list(query_transformers or [])

        model_name = _embedding_models(vectordb).get("dense")
        if dense_embedder is not None:
            self.dense_embedder = dense_embedder
        else:
            if not model_name:
                raise ValueError("VectorDB does not have a dense embedding model configured.")
            model_name = _normalize_dense_model_name(model_name)
            logger.info("Auto-loading dense embedder: %s", model_name)
            self.dense_embedder = EmbedderFactory.get_embedder(model_name)

    def _transform_query(self, query: str) -> str:
        transformed = query
        for transformer in self.query_transformers:
            transformed = transformer.transform(transformed)
        return transformed

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        dense_query = self.dense_embedder.embed_text(self._transform_query(query))
        return list(
            self.vectordb.search_dense(
                query_vector=dense_query,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        namespace, final_filter = self._prepare_filter(filter)

        try:
            candidates = self.retrieve_candidates(
                query,
                top_k=top_k,
                namespace=namespace,
                filter=final_filter if final_filter else None,
            )
            return self._to_search_results(
                candidates,
                source=self.default_source,
                similarity_threshold=similarity_threshold,
            )
        except Exception as e:
            log_msg = f"Semantic search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if final_filter:
                log_msg += f" with filter {final_filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []


class SparseCandidateRetriever(BaseRetriever, CandidateRetriever):
    """Sparse-only candidate retriever, also usable as a standalone retriever."""

    default_source = "sparse"
    source = "sparse"

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        *,
        sparse_embedder: Optional[Any] = None,
    ) -> None:
        super().__init__(vectordb)

        model_name = _embedding_models(vectordb).get("sparse")
        if sparse_embedder is not None:
            self.sparse_embedder = sparse_embedder
        else:
            if not model_name:
                raise ValueError("VectorDB does not have a sparse embedding model configured.")
            logger.info("Auto-loading sparse embedder: %s", model_name)
            self.sparse_embedder = EmbedderFactory.get_sparse_embedder(model_name)

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        sparse_query = self.sparse_embedder.embed_text(query)
        return list(
            self.vectordb.search_sparse(
                query_sparse=sparse_query,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        namespace, final_filter = self._prepare_filter(filter)

        try:
            candidates = self.retrieve_candidates(
                query,
                top_k=top_k,
                namespace=namespace,
                filter=final_filter if final_filter else None,
            )
            return self._to_search_results(
                candidates,
                source=self.default_source,
                similarity_threshold=similarity_threshold,
            )
        except Exception as e:
            log_msg = f"Sparse search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if final_filter:
                log_msg += f" with filter {final_filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []


class HybridCandidateRetriever(BaseRetriever, CandidateRetriever):
    """Hybrid dense+sparse candidate retriever with optional backend acceleration."""

    default_source = "hybrid"
    source = "hybrid"

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        *,
        semantic_retriever: Optional[SemanticCandidateRetriever] = None,
        sparse_retriever: Optional[SparseCandidateRetriever] = None,
        prefetch_k: int = 50,
        fusion: Optional[FusionStrategy] = None,
    ) -> None:
        super().__init__(vectordb)
        self.prefetch_k = prefetch_k
        self.semantic_retriever = semantic_retriever or SemanticCandidateRetriever(vectordb)
        self.sparse_retriever = sparse_retriever or SparseCandidateRetriever(vectordb)
        self.fusion = fusion or RRFFusion()

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.vectordb.capabilities().get("dense") and self.vectordb.capabilities().get("sparse"):
            try:
                dense_query = self.semantic_retriever.dense_embedder.embed_text(query)
                sparse_query = self.sparse_retriever.sparse_embedder.embed_text(query)
                return list(
                    self.vectordb.search_hybrid(
                        dense_query=dense_query,
                        sparse_query=sparse_query,
                        top_k=top_k,
                        prefetch_k=self.prefetch_k,
                        namespace=namespace,
                        filter=filter,
                    )
                )
            except NotImplementedError:
                pass

        dense_results = self.semantic_retriever.retrieve_candidates(
            query,
            top_k=self.prefetch_k,
            namespace=namespace,
            filter=filter,
        )
        sparse_results = self.sparse_retriever.retrieve_candidates(
            query,
            top_k=self.prefetch_k,
            namespace=namespace,
            filter=filter,
        )
        return self.fusion.fuse([dense_results, sparse_results], top_k=top_k)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        namespace, final_filter = self._prepare_filter(filter)

        try:
            candidates = self.retrieve_candidates(
                query,
                top_k=top_k,
                namespace=namespace,
                filter=final_filter if final_filter else None,
            )
            return self._to_search_results(
                candidates,
                source=self.default_source,
                similarity_threshold=similarity_threshold,
            )
        except Exception as e:
            log_msg = f"Hybrid search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if final_filter:
                log_msg += f" with filter {final_filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []
