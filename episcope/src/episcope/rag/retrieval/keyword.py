import logging
from typing import Any, Dict, List, Sequence, Optional

from episcope.rag.interfaces import AbstractRetriever
from episcope.vectordb.base import AbstractVectorDB
from episcope.schemas import SearchResult

logger = logging.getLogger(__name__)

class KeywordRetriever(AbstractRetriever):
    """Performs keyword-based search over documents in a VectorDB."""

    def __init__(self, vectordb: AbstractVectorDB):
        self.vectordb = vectordb
        self._allowed_filter_keys = self.vectordb.get_payload_keys()

    def retrieve(
        self,
        query: str, # space-separated keywords
        *,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """
        Perform keyword-based search.
        The query is a space-separated string of keywords.
        An optional 'filter' dictionary can be used for metadata filtering.
        """
        final_filter = filter.copy() if filter else {}
        namespace = final_filter.pop("paper_id", None) or final_filter.pop("namespace", None)

        for key in final_filter:
            if key not in self._allowed_filter_keys:
                raise ValueError(f"Invalid filter key: {key}. Allowed keys are: {self._allowed_filter_keys}")

        keywords = query.lower().split()
        if not keywords:
            return []

        try:
            # this line gets all chunks for the paper_id (if namespace is paper-scoped)....
            chunks = self.vectordb.get_points(namespace=namespace, filter=final_filter if final_filter else None)
            
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
                    paper_id=chunk.get("paper_id", ""),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=0.0,  # NOT USED IN KEYWORD SEARCH -- kept for compatibility, but should include some WARNING when a keyword retriever is called in a context where similarity_score is expected
                    source="keyword",
                    rank_score=float(score) # Store hit count in rank_score
                ))
            return results

        except Exception as e:
            log_msg = f"Keyword search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if final_filter:
                log_msg += f" with filter {final_filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """
        Perform keyword search for a single query scoped to a specific paper.
        """
        final_filter = filter.copy() if filter else {}
        final_filter["paper_id"] = paper_id
        return self.retrieve(
            query,
            top_k=top_k,
            filter=final_filter,
        )
