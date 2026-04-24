import logging
from typing import Any, Dict, Sequence, Optional

from episcope.rag.retrieval.base import BaseRetriever
from episcope.vectordb.base import AbstractVectorDB
from episcope.schemas import SearchResult

logger = logging.getLogger(__name__)


class KeywordRetriever(BaseRetriever):
    """Performs keyword-based search over documents in a VectorDB."""

    def __init__(self, vectordb: AbstractVectorDB):
        super().__init__(vectordb)
        self.default_source = "keyword"

    def retrieve(
        self,
        query: str,  # space-separated keywords
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """
        Perform keyword-based search.
        The query is a space-separated string of keywords.
        An optional 'filter' dictionary can be used for metadata filtering.
        """
        namespace, final_filter = self._prepare_filter(filter)

        keywords = query.lower().split()
        if not keywords:
            return []

        try:
            # this line gets all chunks for the paper_id (if namespace is paper-scoped)....
            chunks = self.vectordb.get_points(
                namespace=namespace, filter=final_filter if final_filter else None
            )

            hits = []
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                hit_count = sum(text.count(k) for k in keywords)
                if hit_count > 0:
                    hits.append((chunk, hit_count))

            hits.sort(key=lambda x: x[1], reverse=True)

            results = []
            for chunk, score in hits[:top_k]:
                results.append(
                    self._to_search_result(
                        {**chunk, "score": 0.0},
                        source=self.default_source,
                        rank_score=float(score),
                    )
                )
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
