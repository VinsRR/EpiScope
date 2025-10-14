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

    def __init__(self, embedder: SimplifiedEmbedder, vectordb: AbstractVectorDB, hyde: Optional[HYDE] = None):
        self.embedder = embedder
        self.vectordb = vectordb
        self.hyde = hyde

    def retrieve(self, query: str, *, top_k: int = 5, similarity_threshold: float = 0.0, **kwargs: Any) -> Sequence[SearchResult]:
        """Perform semantic search for a single query."""
        namespace = kwargs.get("paper_id") or kwargs.get("namespace")
        if not namespace:
            raise ValueError("paper_id or namespace must be provided for semantic search.")

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
                namespace=namespace
            )

            results = []
            for chunk in chunks:
                score = chunk.get("score", 0.0)
                if score < similarity_threshold:
                    continue

                results.append(SearchResult(
                    id=str(chunk.get("id", "")),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=score,
                    source="semantic"
                ))
            return results

        except Exception as e:
            logger.debug(f"Semantic search failed for query '{query}' on namespace {namespace}: {e}")
            return []