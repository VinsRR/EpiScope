from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, TypeAlias

from episcope.rag.retrieval.base import BaseRetriever
from episcope.schemas import SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.evidence_reranking import (
    EvidenceRerankingStrategy,
    NoOpEvidenceReranker,
)
from episcope.workflows.classification.utils import result_score

CategoryResults: TypeAlias = Dict[str, Dict[str, SearchResult]]


class ClassificationEvidenceSelector:
    """Select evidence for classification via retrieval, deduplication, and optional reranking."""

    def __init__(
        self,
        retriever: BaseRetriever,
        config: BaseClassifierConfig,
        *,
        evidence_reranker: EvidenceRerankingStrategy | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config
        self.evidence_reranker = (evidence_reranker or NoOpEvidenceReranker()).configure(config)

    def select(self, paper_id: str, *, top_k: int) -> List[SearchResult]:
        category_results = self._collect_retrieved_candidates(paper_id, top_k=top_k)
        reranked_results = self.evidence_reranker.rerank(category_results)
        return self._finalize(reranked_results, top_k=top_k)

    def _collect_retrieved_candidates(
        self,
        paper_id: str,
        *,
        top_k: int,
    ) -> CategoryResults:
        category_results: DefaultDict[str, Dict[str, SearchResult]] = defaultdict(dict)
        for category, queries in self.config.template_paragraphs.items():
            for query in queries:
                retrieved = self.retriever.retrieve_by_paper(query, paper_id, top_k=top_k)
                self._merge_category_results(category_results[category], retrieved)
        return dict(category_results)

    def _finalize(
        self,
        category_results: CategoryResults,
        *,
        top_k: int,
    ) -> List[SearchResult]:
        ranked = self._best_labeled_chunks(category_results)
        ranked.sort(key=result_score, reverse=True)
        return ranked[:top_k]

    def _best_labeled_chunks(self, category_results: CategoryResults) -> List[SearchResult]:
        winners: Dict[str, tuple[str, SearchResult]] = {}
        for category, chunks_by_text in category_results.items():
            for text, chunk in chunks_by_text.items():
                current = winners.get(text)
                if current is None or result_score(chunk) > result_score(current[1]):
                    winners[text] = (category, chunk)

        labeled_chunks: List[SearchResult] = []
        for category, chunk in winners.values():
            labeled_chunks.append(self._with_category(chunk, category))
        return labeled_chunks

    def _merge_category_results(
        self,
        destination: Dict[str, SearchResult],
        chunks: Iterable[SearchResult],
    ) -> None:
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            current = destination.get(text)
            if current is None or result_score(chunk) > result_score(current):
                destination[text] = chunk

    @staticmethod
    def _with_category(chunk: SearchResult, category: str) -> SearchResult:
        chunk.artifacts["category"] = category
        return chunk
