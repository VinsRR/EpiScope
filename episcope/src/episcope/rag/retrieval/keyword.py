import logging
from typing import Any, Dict, List, Sequence

from episcope.rag.interfaces import AbstractRetriever
from episcope.vectordb.base import AbstractVectorDB
from episcope.utils.types import SearchResult

logger = logging.getLogger(__name__)

class KeywordRetriever(AbstractRetriever):
    """Performs keyword-based search over documents in a VectorDB."""

    def __init__(self, vectordb: AbstractVectorDB):
        self.vectordb = vectordb

    def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> Sequence[SearchResult]:
        """
        Perform keyword-based search.
        The query is a space-separated string of keywords.
        """
        paper_id = kwargs.get("paper_id")
        if not paper_id:
            raise ValueError("paper_id must be provided for keyword search.")

        keywords = query.lower().split()
        if not keywords:
            return []

        try:
            chunks = self.vectordb.get_points(namespace=paper_id)
            
            hits = []
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                hit_count = sum(text.count(k) for k in keywords)
                if hit_count > 0:
                    hits.append((chunk, hit_count))
            
            hits.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for chunk, score in hits[:top_k]:
                results.append(SearchResult(
                    id=str(chunk.get("id", "")),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=0.0,
                    source="keyword",
                    rank_score=float(score) # Store hit count in rank_score
                ))
            return results

        except Exception as e:
            logger.debug(f"Keyword search failed for paper {paper_id}: {e}")
            return []
