from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, TypeAlias

from episcope.rag.postprocessing.reranker import CrossEncoderReranker, Reranker
from episcope.schemas import SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.utils import result_score

CategoryResults: TypeAlias = Dict[str, Dict[str, SearchResult]]


class EvidenceRerankingStrategy(ABC):
    """Rerank category-scoped evidence candidates before final selection."""

    def configure(self, config: BaseClassifierConfig) -> "EvidenceRerankingStrategy":
        """Bind workflow config when the strategy needs it."""
        return self

    @abstractmethod
    def rerank(self, category_results: CategoryResults) -> CategoryResults:
        """Return reranked evidence grouped by category."""


class NoOpEvidenceReranker(EvidenceRerankingStrategy):
    """Leave retrieval results unchanged."""

    def rerank(self, category_results: CategoryResults) -> CategoryResults:
        return category_results


class BaseCrossEncoderEvidenceReranker(EvidenceRerankingStrategy, ABC):
    """Shared helpers for cross-encoder evidence reranking strategies."""

    def __init__(
        self,
        cross_encoder_reranker: Reranker,
        config: Optional[BaseClassifierConfig] = None,
        *,
        top_k: int | None = None,
    ) -> None:
        self.cross_encoder_reranker = cross_encoder_reranker
        self.config = config
        self.top_k = top_k

    def configure(self, config: BaseClassifierConfig) -> "BaseCrossEncoderEvidenceReranker":
        if self.config is None:
            self.config = config
        return self

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        *,
        config: Optional[BaseClassifierConfig] = None,
        top_k: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> "BaseCrossEncoderEvidenceReranker":
        reranker = CrossEncoderReranker.from_huggingface(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        return cls(
            cross_encoder_reranker=reranker,
            config=config,
            top_k=top_k,
        )

    def _truncate(self, chunks: Iterable[SearchResult]) -> List[SearchResult]:
        ranked = sorted(chunks, key=result_score, reverse=True)
        if self.top_k is None:
            return list(ranked)
        return ranked[:self.top_k]

    def _cross_encoder_query(self, category: str) -> str:
        if self.config is None:
            raise ValueError("Cross-encoder evidence reranker requires a classifier config before use.")
        templates = self.config.cr_template_sentences.get(category) or self.config.template_paragraphs.get(category)
        if not templates:
            raise ValueError(f"No retrieval or cross-encoder templates configured for category {category!r}.")
        return templates[0]

    def _categories(self) -> List[str]:
        if self.config is None:
            raise ValueError("Cross-encoder evidence reranker requires a classifier config before use.")
        categories = list(self.config.cr_template_sentences.keys())
        if categories:
            return categories
        return list(self.config.template_paragraphs.keys())

    def _index_by_text(
        self,
        chunks: Iterable[SearchResult],
        *,
        category: str,
    ) -> Dict[str, SearchResult]:
        indexed: Dict[str, SearchResult] = {}
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            chunk.artifacts["category"] = category
            indexed[text] = chunk
        return indexed

    @staticmethod
    def _clone_chunk(chunk: SearchResult) -> SearchResult:
        copied = deepcopy(chunk)
        if copied.artifacts is None:
            copied.artifacts = {}
        return copied


class WithinLabelCrossEncoderReranker(BaseCrossEncoderEvidenceReranker):
    """Rerank independently within each label/category."""

    def rerank(self, category_results: CategoryResults) -> CategoryResults:
        reranked: CategoryResults = {}
        for category, chunks_by_text in category_results.items():
            if not chunks_by_text:
                continue

            query = self._cross_encoder_query(category)
            candidates = self._truncate(chunks_by_text.values())
            rescored = self.cross_encoder_reranker.rerank(
                query=query,
                results=list(candidates),
                top_k=len(candidates),
            )
            reranked[category] = self._index_by_text(rescored, category=category)
        return reranked


class GlobalCrossEncoderReranker(BaseCrossEncoderEvidenceReranker):
    """Rerank one shared candidate pool against each label/category query."""

    def rerank(self, category_results: CategoryResults) -> CategoryResults:
        shared_candidates = self._truncate(self._best_unique_chunks(category_results).values())
        if not shared_candidates:
            return {}

        reranked: CategoryResults = {}
        for category in self._categories():
            query = self._cross_encoder_query(category)
            rescored = self.cross_encoder_reranker.rerank(
                query=query,
                results=[self._clone_chunk(chunk) for chunk in shared_candidates],
                top_k=len(shared_candidates),
            )
            reranked[category] = self._index_by_text(rescored, category=category)
        return reranked

    @staticmethod
    def _best_unique_chunks(category_results: CategoryResults) -> Dict[str, SearchResult]:
        unique: Dict[str, SearchResult] = {}
        for chunks_by_text in category_results.values():
            for text, chunk in chunks_by_text.items():
                current = unique.get(text)
                if current is None or result_score(chunk) > result_score(current):
                    unique[text] = chunk
        return unique
