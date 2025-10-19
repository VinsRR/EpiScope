import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from episcope.db.academic_db import AcademicDB
from episcope.parse_configs import PaperClassifierConfig
from episcope.pipelines.base import AbstractRAG
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.schemas import (
    ClassificationOutput,
    ClassificationResult,
    PaperMetadata,
    PaperType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperClassifier(AbstractRAG):
    """Multi-modal paper classification using RAG and semantic similarity."""

    def __init__(self, 
                 retriever: AbstractRetriever, 
                 generator: Generator,
                 strategy_name: str = None,
                 config: Optional[PaperClassifierConfig] = None,
                 academic_db: Optional[AcademicDB] = None
    ):
        super().__init__(retriever, generator)
        self.config = config or PaperClassifierConfig()
        self.academic_db = academic_db
        self.template_paragraphs = self.config.template_paragraphs
        self.classification_mapping = self.config.classification_mapping
        self.category_labels = self.config.category_labels
        #
        self.strategy_name = strategy_name

    def run(self, paper_id: str, metadata: Optional[PaperMetadata] = None) -> ClassificationResult:
        """
        Run the classification pipeline for a single paper.
        """
        if self.academic_db:
            metadata = self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        elif not metadata:
            raise ValueError("metadata must be provided when academic_db is not available.")

        relevant_chunks = self.get_relevant_chunks(metadata, paper_id, top_k=self.config.top_k)
        return self.classify_based_on_relevant_chunks(relevant_chunks, metadata)

    def get_relevant_chunks(self, metadata: PaperMetadata, paper_id: str, top_k: int = 10) -> Dict[
        str, List[Tuple[str, float]]]:
        """Main method to retrieve relevant chunks for each paper type."""
        aggregated_chunks = defaultdict(list)
        for paper_type, templates in self.template_paragraphs.items():
            for template_query in templates:
                retrieved_chunks = self.retriever.retrieve_by_paper(template_query, paper_id, top_k=top_k)
                for chunk in retrieved_chunks:
                    aggregated_chunks[paper_type].append((chunk.text, chunk.similarity_score))

        return self._deduplicate_and_rank_chunks(aggregated_chunks, top_k)

    def _deduplicate_and_rank_chunks(self, aggregated_chunks: Dict[str, List[Tuple[str, float]]],
                                    top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """Remove duplicates and keep top-k chunks for each paper type."""
        final_chunks = {}
        for paper_type, chunks in aggregated_chunks.items():
            unique_chunks = {}
            for text, score in chunks:
                text = text.strip()
                if text and (text not in unique_chunks or score > unique_chunks[text]):
                    unique_chunks[text] = score

            sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1], reverse=True)[:top_k] # COULD USE A RERANKER HERE
            final_chunks[paper_type] = sorted_chunks
        return final_chunks

    def classify_based_on_relevant_chunks(self, relevant_chunks: Dict[str, List[Tuple[str, float]]],
                                         metadata: PaperMetadata) -> ClassificationResult:
        """Classify paper based on relevant chunks using LLM."""
        if not any(relevant_chunks.values()):
            logger.warning("No relevant chunks found for classification.")

        return self._llm_classify(metadata, relevant_chunks)

    def _llm_classify(self, metadata: PaperMetadata,
                      relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> ClassificationResult:
        """Perform LLM-based classification."""
        for attempt in range(2):
            try:
                provenance = self.generator.generate(
                    contexts=[],
                    message_builder=self._build_classification_messages,
                    metadata=metadata,
                    relevant_chunks=relevant_chunks,
                    format="json"
                )
                response_content = provenance.answer
                return self._parse_classification_response(response_content)
            except Exception as e:
                logger.warning(f"Classification attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    return self._create_fallback_result()
        return self._create_fallback_result()

    def _format_chunks_for_prompt(self, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> str:
        """Format chunks information for the LLM prompt."""
        chunks_info = ""
        for paper_type, chunks in relevant_chunks.items():
            if not chunks:
                continue
            avg_score = np.mean([score for _, score in chunks]) if chunks else 0
            chunks_info += f"\n{paper_type.upper()} (avg {avg_score:.3f}):\n"
            for text, score in chunks:
                chunks_info += f"  • {score:.3f}: {text}\n"
        return chunks_info

    def _build_classification_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Create the classification prompt."""
        metadata = kwargs.get("metadata")
        relevant_chunks = kwargs.get("relevant_chunks")
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        schema = ClassificationOutput.model_json_schema()
        categories = "\n".join(f"{key} – {value}" for key, value in self.category_labels.items())

        prompt = self.config.prompt_template.format(
            categories=categories,
            title=metadata.title,
            abstract=metadata.abstract or 'N/A',
            keywords=', '.join(metadata.keywords or []),
            chunks_info=chunks_info,
            schema=json.dumps(schema)
        )
        return [{"role": "user", "content": prompt}]

    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        parsed = ClassificationOutput.model_validate_json(response_content)
        paper_type = self.classification_mapping.get(parsed.classification, PaperType.DATA_ANALYSIS)

        class_probs = parsed.class_probabilities or {}
        if parsed.classification in self.category_labels:
            class_probs[self.category_labels[parsed.classification]] = parsed.confidence or 0.9

        for label in self.category_labels.values():
            if label not in class_probs:
                class_probs[label] = 0.01

        confidence = parsed.confidence or class_probs.get(self.category_labels.get(parsed.classification, "Other"),
                                                           0.9)

        return ClassificationResult(
            paper_type=paper_type,
            confidence=float(confidence),
            class_probabilities=class_probs,
            evidence={"reasoning": parsed.reasoning}
        )

    def _create_fallback_result(self) -> ClassificationResult:
        """Create a fallback classification result when LLM fails."""
        return ClassificationResult(
            paper_type=PaperType.DATA_ANALYSIS,
            confidence=0.0,
            class_probabilities={label: 0.5 for label in self.category_labels.values()},
            evidence={"reasoning": "Classification failed, defaulting to data analysis"}
        )
