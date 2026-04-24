from __future__ import annotations

import logging
from typing import List, Optional

from episcope.db.academic_db import AcademicDB
from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.rag.retrieval.base import BaseRetriever
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.base import AbstractRAG
from episcope.workflows.precision_miner.config import (
    FindDataSourcesConfig,
    PrecisionMinerConfig,
)
from episcope.workflows.precision_miner.output import (
    DetailedExtractionResult,
    ExtractionTrace,
)
from episcope.workflows.precision_miner.parsing import PrecisionMinerResponseParser
from episcope.workflows.precision_miner.prompting import PrecisionMinerPromptBuilder
from episcope.workflows.precision_miner.schemas import ExtractionResult

logger = logging.getLogger(__name__)


class PrecisionMiner(AbstractRAG):
    """Configurable extraction workflow over retrieved paper evidence."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Generator,
        strategy_name: Optional[str] = None,
        config: Optional[PrecisionMinerConfig] = None,
        academic_db: Optional[AcademicDB] = None,
    ):
        super().__init__(retriever, generator)
        self.config = config or FindDataSourcesConfig()
        self.academic_db = academic_db
        self.strategy_name = strategy_name
        self.prompt_builder = PrecisionMinerPromptBuilder(self.config)
        self.response_parser = PrecisionMinerResponseParser()

    def run(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata] = None,
    ) -> ExtractionResult:
        """Run extraction and return the compact structured result."""
        detailed = self.run_detailed(paper_id, metadata=metadata)
        return detailed.result

    def run_detailed(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata] = None,
    ) -> DetailedExtractionResult:
        """Run extraction and return the detailed result with provenance and trace."""
        metadata = self._resolve_metadata(paper_id, metadata)
        relevant_chunks = self.retrieve_chunks(paper_id)
        messages = self.prompt_builder.build_messages(metadata, relevant_chunks)

        try:
            provenance = self.generator.generate(
                contexts=relevant_chunks,
                message_builder=lambda **_: messages,
                format="json",
            )
            result = self.response_parser.parse(provenance.answer)
            raw_response = provenance.answer
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            result = self.response_parser.fallback(exc)
            raw_response = ""
            provenance = Provenance(answer=raw_response, evidences=[])

        return DetailedExtractionResult(
            paper_id=paper_id,
            metadata=metadata,
            result=result,
            provenance=provenance,
            trace=ExtractionTrace(
                prompt_messages=messages,
                raw_llm_response=raw_response,
            ),
            relevant_chunks=list(relevant_chunks),
        )

    def _resolve_metadata(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata],
    ) -> PaperMetadata:
        if self.academic_db:
            return self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        if metadata is None:
            raise ValueError(
                "metadata must be provided when academic_db is not available."
            )
        return metadata

    def retrieve_chunks(self, paper_id: str) -> List[SearchResult]:
        """Retrieve and deduplicate relevant chunks for the extraction task."""
        all_chunks: List[SearchResult] = []
        for query in self.config.retrieval_templates:
            retrieved = self.retriever.retrieve_by_paper(
                query,
                paper_id,
                top_k=self.config.top_k,
            )
            all_chunks.extend(retrieved)
        filtered = self._filter_chunks(all_chunks)
        return self._deduplicate_chunks(filtered)

    def _filter_chunks(self, chunks: List[SearchResult]) -> List[SearchResult]:
        if not self.config.section_filters:
            return chunks

        allowed = {section.lower() for section in self.config.section_filters}
        filtered: List[SearchResult] = []
        for chunk in chunks:
            section = (chunk.section_type or "").lower()
            if section in allowed:
                filtered.append(chunk)
        return filtered

    def _deduplicate_chunks(self, chunks: List[SearchResult]) -> List[SearchResult]:
        best_by_text = {}
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue

            current = best_by_text.get(text)
            if current is None or self._score(chunk) > self._score(current):
                best_by_text[text] = chunk

        ranked = sorted(best_by_text.values(), key=self._score, reverse=True)
        return ranked[: self.config.top_k]

    @staticmethod
    def _score(chunk: SearchResult) -> float:
        if getattr(chunk, "rank_score", None) is not None:
            return chunk.rank_score
        if getattr(chunk, "similarity_score", None) is not None:
            return chunk.similarity_score
        return 0.0
