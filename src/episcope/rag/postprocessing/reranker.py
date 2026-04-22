from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from episcope.schemas import SearchResult
# from episcope.rag.postprocessing.scoring import Scorer, default_scorer

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Base class for all rerankers.

    `query` is part of the interface even for score-based rerankers that
    ignore it, so every Reranker is drop-in composable inside Retriever
    and CascadeReranker.
    """

    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        ...


# # ---------------------------------------------------------------------------
# # Heuristic reranker
# # ---------------------------------------------------------------------------

# class ResultRanker(Reranker):
#     """Deduplicates and ranks results using an injected Scorer.

#     Does not use `query` — scores are derived purely from result content
#     and pre-computed similarity scores. Use as the first stage of a
#     CascadeReranker to cheaply reduce a large candidate pool.
#     """

#     def __init__(self, scorer: Optional[Scorer] = None):
#         self.scorer = scorer or default_scorer()

#     def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
#         if not results:
#             return []
#         self._apply_scores(results)
#         unique = self._deduplicate(results)
#         unique.sort(key=lambda r: r.rank_score, reverse=True)
#         return unique[:top_k]

#     def _apply_scores(self, results: List[SearchResult]) -> None:
#         for result in results:
#             result.rank_score = self.scorer(result)

#     @staticmethod
#     def _deduplicate(results: List[SearchResult]) -> List[SearchResult]:
#         """Keep the highest-scoring result per normalised text signature."""
#         seen: Dict[str, SearchResult] = {}
#         for result in results:
#             sig = result.artifacts.get("normalized_text", result.text)[:200].lower().strip()
#             if sig not in seen or result.rank_score > seen[sig].rank_score:
#                 seen[sig] = result
#         return list(seen.values())





# ---------------------------------------------------------------------------
# Neural cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker(Reranker):
    """Neural reranker using a cross-encoder model (e.g. Jina, BGE, Cohere).

    Scores (query, passage) pairs jointly — far more accurate than embedding
    similarity alone, but also more expensive. Pair with a CascadeReranker
    to limit the model call to a pre-filtered candidate set.

    Compatible with any model that exposes one of:
      - `.predict(pairs: List[Tuple[str, str]])` — sentence-transformers CrossEncoder
      - `.rank(query: str, passages: List[str])` — FlagEmbedding / Jina reranker API

    Example (Jina via sentence-transformers):
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")
        reranker = CrossEncoderReranker(model)

    Example (FlagEmbedding):
        from FlagEmbedding import FlagReranker
        model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        reranker = CrossEncoderReranker(model)
    """

    def __init__(self, model: Any, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not results:
            return []

        texts = [r.text for r in results]
        scores = self._score(query, texts)

        for result, score in zip(results, scores):
            result.rank_score = float(score)

        results.sort(key=lambda r: r.rank_score, reverse=True)
        return results[:top_k]

    def _score(self, query: str, passages: List[str]) -> List[float]:
        """Normalise across the two most common cross-encoder APIs."""
        # sentence-transformers CrossEncoder: .predict([(query, passage), ...])
        if hasattr(self.model, "predict"):
            pairs = [(query, p) for p in passages]
            return list(self.model.predict(pairs, batch_size=self.batch_size))

        # FlagEmbedding / Jina: .rank(query, passages) -> [{corpus_id, score}, ...]
        # Results come back sorted; we re-align to original order by corpus_id.
        if hasattr(self.model, "rank"):
            ranked = self.model.rank(query, passages, batch_size=self.batch_size)
            score_map: Dict[int, float] = {item["corpus_id"]: item["score"] for item in ranked}
            return [score_map.get(i, 0.0) for i in range(len(passages))]

        raise TypeError(
            f"Model {type(self.model).__name__!r} has neither `.predict` nor `.rank`. "
            "Wrap it in a thin adapter that exposes one of these methods."
        )


    @classmethod
    def from_huggingface(cls, model_name: str, device: Optional[str] = None, **kwargs) -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name, device=device)
        return cls(model, **kwargs)

# ---------------------------------------------------------------------------
# Cascade reranker
# ---------------------------------------------------------------------------

class CascadeReranker(Reranker):
    """Runs rerankers in sequence, each operating on the previous stage's output.

    The canonical setup for data-availability retrieval:

        CascadeReranker([
            (ResultRanker(scorer=data_availability_scorer()), 50),  # cheap: N → 50
            (CrossEncoderReranker(model),                    None), # neural: 50 → top_k
        ])

    Each stage is a (reranker, k) pair:
      - k controls how many results are forwarded to the next stage.
      - Pass k=None to forward all results from that stage unchanged.
      - The final stage always uses the top_k passed to .rerank().
    """

    def __init__(self, stages: List[Tuple[Reranker, Optional[int]]]):
        if not stages:
            raise ValueError("CascadeReranker requires at least one stage.")
        self.stages = stages

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        current = results
        for i, (reranker, stage_k) in enumerate(self.stages):
            is_last = i == len(self.stages) - 1
            k = top_k if is_last else (stage_k or len(current))
            current = reranker.rerank(query, current, k)
        return current
