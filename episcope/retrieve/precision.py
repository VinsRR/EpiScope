"""Retriever implementation for the PrecisionMiner pipeline.

The PrecisionMinerRetriever orchestrates a per‑paper retrieval and
extraction pipeline.  Given a query and one or more target papers,
it retrieves relevant contexts from a per‑paper index, classifies
the paper type, extracts structured data sources using a simple
heuristic extractor, and returns a harmonised result for each
paper.  This retriever is deliberately lightweight to avoid
dependencies on heavy LLM and FAISS libraries while still
demonstrating the expected behaviour of the PrecisionMiner
workflow.

Actual classification and extraction logic can be replaced by
injections of :class:`PaperClassifier` and :class:`LLMExtractor`
instances when full models are available.  In the absence of such
dependencies, basic string heuristics are used.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import AbstractIndexer, AbstractRetriever
from ..core.blueprints.data_blueprints import PaperType, DataSource, ExtractionResult

logger = logging.getLogger(__name__)


class PrecisionMinerRetriever(AbstractRetriever):
    """Specialised retriever for the PrecisionMiner pipeline.

    This retriever expects a per‑paper indexer that supports
    ``index_paper`` and ``search`` methods.  It optionally accepts
    externally supplied classification and extraction callables to
    support heavy LLM‑powered implementations.  When those
    callables are not provided, fallback heuristics are used.
    """

    def __init__(
        self,
        indexer: AbstractIndexer,
        classifier_fn: Optional[Any] = None,
        extractor_fn: Optional[Any] = None,
    ) -> None:
        self.indexer = indexer
        self.classifier_fn = classifier_fn
        self.extractor_fn = extractor_fn

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Sequence[Dict[str, Any]]:
        """Retrieve structured results for the given query across papers.

        Args:
            query: The user query to run against each paper.
            top_k: Number of contexts to retrieve per paper.
            paper_ids: A list of paper identifiers to restrict retrieval.
            **kwargs: Unused in the fallback implementation.

        Returns:
            A list of result dictionaries, one per paper.  Each result
            contains the paper_id, classification, extracted data
            sources and the raw contexts retrieved.
        """
        results: List[Dict[str, Any]] = []
        target_papers = paper_ids or list(self.indexer._embeddings.keys())  # type: ignore[attr-defined]
        for pid in target_papers:
            # Retrieve contexts for this paper
            contexts = [c for c in self.indexer.search(query, top_k=top_k) if c.get("paper_id") == pid]
            if not contexts:
                continue
            # Determine paper type
            classification = self._classify_paper(contexts)
            # Extract data sources
            extraction = self._extract_data_sources(contexts, classification, query)
            results.append({
                "paper_id": pid,
                "classification": classification.value,
                "data_sources_description": extraction.description,
                "data_sources": [ds.__dict__ for ds in extraction.data_sources],
                "references": [ref.to_dict() for ref in extraction.references],
                "contexts": contexts,
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_paper(self, contexts: Sequence[Dict[str, Any]]) -> PaperType:
        """Classify a paper based on retrieved contexts.

        If an external classifier function was provided at
        construction time, it will be used.  Otherwise a simple
        keyword heuristic is applied: if any context contains the
        word 'review' (case‑insensitive) the paper is considered a
        literature review, otherwise it is classified as data
        analysis.  Future extensions can incorporate more labels.
        """
        if self.classifier_fn is not None:
            try:
                return self.classifier_fn(contexts)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.warning(f"Classifier function failed: {exc}; falling back to heuristic")
        for ctx in contexts:
            content = (ctx.get("content") or ctx.get("text") or "").lower()
            if "review" in content and "data" not in content:
                return PaperType.LITERATURE_REVIEW
        return PaperType.DATA_ANALYSIS

    def _extract_data_sources(
        self, contexts: Sequence[Dict[str, Any]], paper_type: PaperType, query: str
    ) -> ExtractionResult:
        """Extract structured data sources from contexts.

        When an external extractor function is supplied at
        construction time it will be invoked.  Otherwise this method
        searches for mentions of datasets or supplementary material
        using a simple regex and constructs a :class:`ExtractionResult`.
        """
        if self.extractor_fn is not None:
            try:
                return self.extractor_fn(contexts, paper_type, query)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.warning(f"Extractor function failed: {exc}; falling back to heuristic")
        # Heuristic: find lines containing 'data' or 'supplement'
        found: List[DataSource] = []
        for ctx in contexts:
            content = (ctx.get("content") or ctx.get("text") or "")
            for line in content.split("\n"):
                lower = line.lower()
                if "data" in lower or "supplement" in lower:
                    name = "Supplementary Data" if "supplement" in lower else "Dataset"
                    found.append(DataSource(source_name=name, explanation=line.strip(), section_found=ctx.get("section_title", "")))
        desc = "No data sources mentioned." if not found else f"Found {len(found)} data source(s)."
        return ExtractionResult(description=desc, data_sources=found, references=[])
