from __future__ import annotations

from typing import Any, Dict, List, Sequence

from episcope.rag.retrieval.components import FusionStrategy


class RRFFusion(FusionStrategy):
    """Reciprocal rank fusion over multiple candidate lists."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        results_lists: Sequence[Sequence[Dict[str, Any]]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        fused_scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for results in results_lists:
            for rank, item in enumerate(results):
                doc_id = item.get("id")
                if doc_id is None:
                    continue

                doc_id = str(doc_id)
                if doc_id not in items:
                    items[doc_id] = dict(item)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results: List[Dict[str, Any]] = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            item = dict(items[doc_id])
            item["rrf_score"] = rrf_score
            fused_results.append(item)

        return fused_results
