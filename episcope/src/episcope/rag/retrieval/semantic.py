import logging
from typing import Any, Dict, Optional, Sequence

from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.rag.interfaces import AbstractRetriever
from episcope.utils.hyde import HYDE
from episcope.vectordb.base import AbstractVectorDB
from episcope.utils.types import SearchResult

logger = logging.getLogger(__name__)

class SemanticRetriever(AbstractRetriever):
    """Performs semantic search using a vector database, with optional HYDE."""

    def __init__(self, vectordb: AbstractVectorDB, hyde: Optional[HYDE] = None):
        self.vectordb = vectordb
        embedder = SimplifiedEmbedder(embed_model=vectordb.get_embedding_model())
        self.embedder = embedder
        self.hyde = hyde
        self._allowed_filter_keys = self.vectordb.get_payload_keys()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Sequence[SearchResult]:
        """
        Perform semantic search for a single query.
        A 'paper_id' or 'namespace' can be passed in kwargs to scope the search.
        An optional 'filter' dictionary can be used for more specific metadata filtering.
        """
        namespace = kwargs.get("paper_id") or kwargs.get("namespace")

        if filter:
            for key in filter:
                if key not in self._allowed_filter_keys:
                    raise ValueError(f"Invalid filter key: {key}. Allowed keys are: {self._allowed_filter_keys}")

        final_query = query
        if self.hyde:
            hypothetical_doc = self.hyde.generate(query)
            if hypothetical_doc and "failed" not in hypothetical_doc.lower():
                final_query = f"{query}\n\n{hypothetical_doc}"

        try:
            query_embedding = self.embedder.embed_text(final_query)
            chunks = self.vectordb.search(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )

            results = []
            for chunk in chunks:
                score = chunk.get("score", 0.0)
                if score < similarity_threshold:
                    continue

                results.append(
                    SearchResult(
                        id=str(chunk.get("id", "")),
                        text=chunk.get("text", ""),
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=score,
                        source="semantic",
                    )
                )
            return results

        except Exception as e:
            log_msg = f"Semantic search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if filter:
                log_msg += f" with filter {filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        **kwargs: Any,
    ) -> Sequence[SearchResult]:
        """
        Perform semantic search for a single query scoped to a specific paper.
        """
        return self.retrieve(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter={"paper_id": paper_id},
            **kwargs,
        )