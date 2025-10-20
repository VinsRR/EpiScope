"""Retriever implementation for the PrecisionMiner pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from episcope.rag.interfaces import AbstractRetriever
from episcope.schemas import PaperType, DataSource, ExtractionResult
from episcope.vectordb.base import AbstractVectorDB
from episcope.rag.embeddings.factory import EmbedderFactory

logger = logging.getLogger(__name__)

class PrecisionMinerRetriever(AbstractRetriever):
    """Specialised retriever for the PrecisionMiner pipeline."""

    def __init__(
        self,
        db: AbstractVectorDB,
        classifier_fn: Optional[Any] = None,
        extractor_fn: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.classifier_fn = classifier_fn
        self.extractor_fn = extractor_fn

        model_name = self.db.get_embedding_model()
        if not model_name:
            raise ValueError("VectorDB does not have an embedding model configured.")

        logger.info(f"VectorDB is configured with embedding model: '{model_name}'. Instantiating corresponding embedder for PrecisionMinerRetriever.")
        self.embedder = EmbedderFactory.get_embedder(model_name)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Sequence[Dict[str, Any]]:
        """Retrieve structured results for the given query across papers."""
        results: List[Dict[str, Any]] = []
        if not paper_ids:
            logger.warning("PrecisionMinerRetriever requires a list of paper_ids to search.")
            return []

        query_vector = self.embedder.embed_text(query)

        for pid in paper_ids:
            contexts = self.db.search(query_vector, top_k=top_k, namespace=pid)
            if not contexts:
                continue
            
            classification = self._classify_paper(contexts)
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

    def _classify_paper(self, contexts: Sequence[Dict[str, Any]]) -> PaperType:
        """Classify a paper based on retrieved contexts."""
        if self.classifier_fn is not None:
            try:
                return self.classifier_fn(contexts)
            except Exception as exc:
                logger.warning(f"Classifier function failed: {exc}; falling back to heuristic")
        for ctx in contexts:
            content = (ctx.get("content") or ctx.get("text") or "").lower()
            if "review" in content and "data" not in content:
                return PaperType.LITERATURE_REVIEW
        return PaperType.DATA_ANALYSIS

    def _extract_data_sources(
        self,
        contexts: Sequence[Dict[str, Any]],
        paper_type: PaperType,
        query: str,
    ) -> ExtractionResult:
        """Extract structured data sources from contexts."""
        if self.extractor_fn is not None:
            try:
                return self.extractor_fn(contexts, paper_type, query)
            except Exception as exc:
                logger.warning(f"Extractor function failed: {exc}; falling back to heuristic")
        
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
