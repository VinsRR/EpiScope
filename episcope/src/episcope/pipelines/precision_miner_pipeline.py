import json
import logging
from typing import Dict, List, Optional

from episcope.db.academic_db import AcademicDB
from episcope.parse_configs import PrecisionMinerConfig, FindDataSourcesConfig
from episcope.pipelines.base import AbstractRAG
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.schemas import (
    ExtractionResult,
    PaperMetadata,
    ExtractionItem,
    ExtractionResultSchema,
)

logger = logging.getLogger(__name__)


class PrecisionMiner(AbstractRAG):
    """A configurable RAG pipeline for extracting structured information from papers."""

    def __init__(self, 
                 retriever: AbstractRetriever, 
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
        Run the extraction pipeline for a single paper.
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
        for attempt in range(2):
            try:
                provenance = self.generator.generate(
                    contexts=[],
                    message_builder=self._build_extraction_messages,
                    metadata=metadata,
                    relevant_chunks=relevant_chunks,
                    format="json"
                )
                response_content = provenance.answer
                return response_content #self._parse_extraction_response(response_content)
            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    return ExtractionResult(description="Extraction failed", items=[])
        return ExtractionResult(description="Extraction failed", items=[])

    def _build_extraction_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Build the prompt for the extraction task."""
        metadata = kwargs.get("metadata")
        relevant_chunks = kwargs.get("relevant_chunks")
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        schema = ExtractionResultSchema.model_json_schema()

        prompt = self.config.prompt_template.format(
            title=metadata.title,
            abstract=metadata.abstract or 'N/A',
            keywords=', '.join(metadata.keywords or []),
            chunks_info=chunks_info,
            schema=json.dumps(schema)
        )
        print(prompt)
        return [{"role": "user", "content": prompt}]

    def _format_chunks_for_prompt(self, relevant_chunks: List[Dict]) -> str:
        """Format chunks information for the LLM prompt."""
        return "\n\n".join([chunk.text for chunk in relevant_chunks])

    def _parse_extraction_response(self, response: str) -> ExtractionResult:
        """Parse JSON response from extraction LLM."""
        try:
            validated_data = ExtractionResultSchema.model_validate_json(response)
            items = [ExtractionItem(**item.model_dump()) for item in validated_data.items]
            return ExtractionResult(description=validated_data.description, items=items)
        except Exception as e:
            logger.error(f"Extraction parsing/validation failed: {e}")
            return ExtractionResult(description=f"Parsing/validation failed: {e}", items=[])

