import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from episcope.db.academic_db import AcademicDB
from episcope.workflows.classification.config import BaseClassifierConfig, PaperTypeClassifierConfig
from episcope.workflows.base import AbstractRAG
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.schemas import PaperMetadata
from .schemas import ClassificationResult 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperClassifier(AbstractRAG):
    """Multi-modal paper classification using RAG and semantic similarity."""

    def __init__(self, 
                 retriever: AbstractRetriever, 
                 generator: Generator,
                 strategy_name: str = None,
                 config: Optional[BaseClassifierConfig] = None,
                 academic_db: Optional[AcademicDB] = None
    ):
        super().__init__(retriever, generator)
        self.config = config or PaperTypeClassifierConfig()
        self.academic_db = academic_db
        self.template_paragraphs = self.config.template_paragraphs
        self.classification_mapping = self.config.classification_mapping
        self.category_labels = self.config.category_labels
        self.strategy_name = strategy_name

    def run(self, paper_id: str, metadata: Optional[PaperMetadata] = None) -> ClassificationResult:
        """
        Run the classification workflow for a single paper.
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

            sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1], reverse=True)[:top_k]
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
        """Perform LLM-based classification with self-correction."""
        messages = self._build_initial_prompt(metadata, relevant_chunks)
        
        for attempt in range(self.config.max_validation_retries):
            try:
                provenance = self.generator.generate(
                    contexts=list(relevant_chunks.values()),
                    message_builder=lambda **kwargs: messages,
                    format="json"
                )
                response_content = provenance.answer
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                continue

            try:
                return self._parse_classification_response(response_content)
            except Exception as e:
                logger.warning(f"Classification parsing attempt {attempt + 1} failed: {e}")
                error_message = f"The JSON output is invalid. Please fix it. Error: {e}"
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": error_message})
        print("\n\nresponse_content:", response_content,"\n\n")
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

    def _build_initial_prompt(self, metadata: PaperMetadata, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> List[Dict[str, str]]:
        """Create the initial classification prompt."""
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        schema = self.config.output_schema.model_json_schema()
        categories = "\n".join(f"{key} – {value}" for key, value in self.category_labels.items())

        user_prompt = self.config.user_prompt_template.format(
            categories=categories,
            title=metadata.title,
            abstract=metadata.abstract or 'N/A',
            keywords=', '.join(metadata.keywords or []),
            chunks_info=chunks_info,
            schema=json.dumps(schema)
        )
        system_prompt = self.config.system_prompt.format(
            n_categories=len(self.category_labels),
            category_labels=", ".join(list(self.category_labels.values()))
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        parsed = self.config.output_schema.model_validate_json(response_content)
        classification = self.classification_mapping.get(parsed.classification, self.config.default_classification)

        confidence = parsed.confidence if parsed.confidence is not None else 0.0
        class_probs = parsed.class_probabilities or {}

        return ClassificationResult(
            classification=classification,
            confidence=float(confidence),
            class_probabilities=class_probs,
            evidence={"reasoning": parsed.reasoning}
        )

    def _create_fallback_result(self) -> ClassificationResult:
        """Create a fallback classification result when LLM fails."""
        return ClassificationResult(
            classification=self.config.default_classification,
            confidence=0.0,
            class_probabilities={label: 1/len(self.category_labels) for label in self.category_labels.values() if self.category_labels},
            evidence={"reasoning": "Classification failed, defaulting to unclear"}
        )
