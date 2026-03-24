import json
import logging
from typing import Dict, List, Optional

from episcope.db.academic_db import AcademicDB
from episcope.workflows.precision_miner.config import PrecisionMinerConfig, FindDataSourcesConfig
from episcope.workflows.base import AbstractRAG
from episcope.rag.generation.base import Generator
from episcope.rag.retrieval.base import BaseRetriever
from episcope.schemas import PaperMetadata
from .schemas import ExtractionResult, ExtractionItem, ExtractionResultSchema

logger = logging.getLogger(__name__)


class PrecisionMiner(AbstractRAG):
    """A configurable RAG workflow for extracting structured information from papers."""

    def __init__(self, 
                 retriever: BaseRetriever, 
                 generator: Generator,
                 strategy_name: str = None,
                 config: Optional[PrecisionMinerConfig] = None,
                 academic_db: Optional[AcademicDB] = None):
        super().__init__(retriever, generator)
        self.config = config or FindDataSourcesConfig()
        self.academic_db = academic_db
        self.strategy_name = strategy_name

    def run(self, paper_id: str, metadata: Optional[PaperMetadata] = None) -> ExtractionResult:
        """
        Run the extraction workflow for a single paper.
        """
        if self.academic_db:
            metadata = self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        elif not metadata:
            raise ValueError("metadata must be provided when academic_db is not available.")

        relevant_chunks = self.retrieve_chunks(paper_id)
        return self.generate_extraction(relevant_chunks, metadata)

    def retrieve_chunks(self, paper_id: str) -> List[Dict]:
        """Retrieve relevant chunks for the extraction task."""
        all_chunks = []
        for query in self.config.retrieval_templates:
            chunks = self.retriever.retrieve_by_paper(
                query,
                paper_id,
                top_k=self.config.top_k,
                filter={"section_type": self.config.section_filters} if self.config.section_filters else None,
            )
            all_chunks.extend(chunks)
        return self._deduplicate_chunks(all_chunks)

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Deduplicate a list of chunk dictionaries based on their text content."""
        unique_chunks = {}
        for chunk in chunks:
            if chunk.text not in unique_chunks:
                unique_chunks[chunk.text] = chunk
        return list(unique_chunks.values())

    def generate_extraction(self, relevant_chunks: List[Dict], metadata: PaperMetadata) -> ExtractionResult:
        """Generate the structured extraction using the LLM."""
        try:
            provenance = self.generator.generate(
                contexts=relevant_chunks,
                message_builder=self._build_initial_prompt,
                metadata=metadata,
                format="json"
            )
            response_content = provenance.answer
            return self._parse_extraction_response(response_content)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(description="Extraction failed", items=[])

    def _build_initial_prompt(self, **kwargs) -> List[Dict[str, str]]:
        """Build the prompt for the extraction task."""
        metadata = kwargs.get("metadata")
        contexts = kwargs.get("contexts")
        chunks_info = self._format_chunks_for_prompt(contexts)
        schema = ExtractionResultSchema.model_json_schema()

        user_prompt = self.config.user_prompt_template.format(
            title=metadata.title,
            abstract=metadata.abstract or 'N/A',
            keywords=', '.join(metadata.keywords or []),
            chunks_info=chunks_info,
            schema= json.dumps(schema)
        )
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_chunks_for_prompt(self, relevant_chunks: List[Dict]) -> str:
        """Format chunks information for the LLM prompt."""
        return "\n\n".join([chunk.text for chunk in relevant_chunks])

    def _parse_extraction_response(self, response: str) -> ExtractionResult:
        """Parse JSON response from extraction LLM."""
        try:
            # if response starts with ```json\n it means the LLM formatted it as a string code block
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()

            return ExtractionResult.model_validate_json(response)
        except Exception as e:
            logger.error(f"Extraction parsing/validation failed: {e}")
            return ExtractionResult(description=f"Parsing/validation failed: {e}", items=[])
