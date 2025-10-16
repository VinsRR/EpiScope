import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from episcope.rag.generation.base import Generator
from episcope.utils.data_blueprints import (
    ExtractionResult,
    PaperType,
    Reference,
    StructuredSection,
    DataSource,
)
from episcope.utils.data_blueprints import PaperMetadata

logger = logging.getLogger(__name__)


class DataExtractor:
    """Enhanced LLM extractor using structured GROBID data with integrated classification."""

    def __init__(self, classifier: PaperClassifier, generator: Generator):
        self.classifier = classifier
        self.generator = generator

    def analyze_and_extract(self, sections: List[StructuredSection], metadata: PaperMetadata,
                           references: List[Reference], relevant_chunks: List[Dict],
                           query: str, paper_id: str) -> Tuple[PaperType, ExtractionResult, Optional[Dict]]:
        """Complete pipeline: classify paper type and extract relevant information."""

        logger.info("Classifying paper type...")
        classification_result = self.classifier.run(metadata, paper_id)
        paper_type = classification_result.paper_type

        logger.info(f"Paper classified as: {paper_type.value} (confidence: {classification_result.confidence:.2f})")

        extraction_result, confidence_scores = self._extract_by_paper_type(
            paper_type, relevant_chunks, references, metadata, query, sections
        )

        confidence_scores = confidence_scores or {}
        confidence_scores.update({
            "classification_type": paper_type.value,
            "classification_confidence": classification_result.confidence,
            "classification_evidence": classification_result.evidence
        })

        return paper_type, extraction_result, confidence_scores

    def _extract_by_paper_type(self, paper_type: PaperType, chunks: List[Dict], references: List[Reference],
                              metadata: PaperMetadata, query: str, sections: List[StructuredSection]) -> \
            Tuple[ExtractionResult, Dict]:
        """Route extraction based on paper type."""
        method = {
            PaperType.DATA_ANALYSIS: self._extract_direct_sources,
            PaperType.LITERATURE_REVIEW: self._extract_literature_sources,
        }.get(paper_type, self._extract_direct_sources)

        logger.info(f"Using extraction method for {paper_type.value}")
        return method(chunks, references, metadata, query, sections)

    def _execute_extraction(self, prompt: str, chunks: List[Dict], is_references: bool = False) -> Tuple[
        ExtractionResult, Dict]:
        """Execute the extraction with error handling."""
        try:
            provenance = self.generator.generate(contexts=[], question=prompt, format="json")
            response_text = provenance.answer
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            return self._parse_extraction_response(response_text, is_references)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(description="Extraction failed"), {}

    def _parse_extraction_response(self, response: str, is_references: bool = False) -> Tuple[ExtractionResult, Dict]:
        """Parse JSON response from extraction LLM."""
        try:
            data = json.loads(response)
            description = data.get('data_sources_description', 'No description provided')

            data_sources = [DataSource(**ds_data) for ds_data in data.get('data_sources', []) if
                            isinstance(ds_data, dict)]
            references = [Reference(raw_text=ref_text.strip()) for ref_text in data.get('references', []) if
                          isinstance(ref_text, str)]

            result = ExtractionResult(description=description, data_sources=data_sources, references=references)
            confidence_scores = {"extraction_method": "llm_based", "json_parsing_success": True,
                                 "num_data_sources": len(data_sources), "num_references": len(references)}

            return result, confidence_scores
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return ExtractionResult(description=f"JSON parsing failed: {e}"), {"json_parsing_success": False}
        except Exception as e:
            logger.error(f"Extraction parsing failed: {e}")
            return ExtractionResult(description=f"Parsing failed: {e}"), {"parsing_success": False}

    # Other methods like _get_section_text, _compile_chunk_text, _create_..._prompt should be included here
    # For brevity, they are omitted but assumed to be part of the class.
    def _get_section_text(self, sections: List[StructuredSection], section_types: List[str]) -> str:
        """Extract text from specific section types."""

        section_texts = []
        for section in sections:
            for section_type in section_types:
                if section_type.lower() in section.section_type.lower():
                    section_texts.append(f"=== {section.section_type} ===\n{section.text}\n")
                    break

        return "\n".join(section_texts)

    def _compile_chunk_text(self, chunks: List[Dict], preferred_sections: List[str]) -> str:
        """Compile text from relevant chunks with structure preservation."""

        sections_text = []
        for section_type in preferred_sections:
            section_chunks = [chunk for chunk in chunks if chunk.get('section_type') == section_type]
            if section_chunks:
                section_text = f"\n=== {section_type} (from chunks) ===\n"
                section_text += "\n\n".join(chunk['text'] for chunk in section_chunks)
                sections_text.append(section_text)

        # Fallback to top chunks if no preferred sections found
        if not sections_text:
            sections_text = [chunk['text'] for chunk in chunks[:5]]

        return "\n".join(sections_text)

    def _create_direct_sources_prompt(self, metadata: PaperMetadata, combined_text: str, ref_context: str) -> str:
        """Create prompt for direct data source extraction."""

        return f"""Extract data sources from this epidemiological research paper that conducts original data analysis. 

            Paper Title: {metadata.title}
            Abstract: {metadata.abstract or 'Not available'}
            Keywords: {', '.join(metadata.keywords or [])}

            Structured Methods and Data Sections:
            {combined_text}

            Potential Data Source References:
            {ref_context}

            INSTRUCTIONS:
            Focus on identifying:
            1. Primary datasets, databases, or surveys used
            2. Data collection methods and sources
            3. Population or sample details
            4. Any links or availability information for the data

            Return ONLY valid JSON in this format:
            {{
                "data_sources_description": "Brief description of the primary data sources used for analysis",
                "data_sources": [
                    {{
                        "source_name": "Name of dataset/database/survey",
                        "url": "URL if mentioned or 'N/A'",
                        "explanation": "Brief explanation of how this source was used",
                        "section_found": "Section where this was mentioned"
                    }}
                ]
            }}

            JSON:"""

    def _extract_direct_sources(self, chunks: List[Dict], references: List[Reference], metadata: PaperMetadata,
                               query: str, sections: List[StructuredSection]) -> Tuple[ExtractionResult, Dict]:
        """Extract direct data sources for empirical studies."""

        # Get relevant text from structured sections and chunks
        methods_text = self._get_section_text(sections, ['Methods', 'Data', 'Study Design', 'Participants'])
        chunk_text = self._compile_chunk_text(chunks, ['Methods', 'Data'])
        combined_text = f"{methods_text}\n\n=== RELEVANT EXCERPTS ===\n{chunk_text}"

        # Get data source references
        data_references = [ref for ref in references if ref.is_data_source]
        ref_context = "\n".join([f"- {ref.raw_text}" for ref in data_references])

        prompt = self._create_direct_sources_prompt(metadata, combined_text, ref_context)

        return self._execute_extraction(prompt, chunks, is_references=False)

    def _extract_literature_sources(self, chunks: List[Dict], references: List[Reference], metadata: PaperMetadata,
                                   query: str, sections: List[StructuredSection]) -> Tuple[ExtractionResult, Dict]:
        """Extract literature sources for reviews and meta-analyses."""

        # Get relevant sections
        methods_text = self._get_section_text(sections, ['Methods', 'Literature Search', 'Search Strategy'])
        results_text = self._get_section_text(sections, ['Results', 'Findings', 'Studies Included'])
        chunk_text = self._compile_chunk_text(chunks, ['Methods', 'Results', 'Introduction'])

        combined_text = f"{methods_text}\n\n{results_text}\n\n=== RELEVANT EXCERPTS ===\n{chunk_text}"
        ref_context = "\n".join([f"- {ref.raw_text}" for ref in references[:20]])  # Limit references

        prompt = self._create_literature_sources_prompt(metadata, combined_text, ref_context)

        return self._execute_extraction(prompt, chunks, is_references=True)

    def _create_literature_sources_prompt(self, metadata: PaperMetadata, combined_text: str, ref_context: str) -> str:
        """Create prompt for literature source extraction."""

        return f"""Extract key literature sources from this systematic review/meta-analysis paper.

            Paper Title: {metadata.title}
            Abstract: {metadata.abstract or 'Not available'}
            Keywords: {', '.join(metadata.keywords or [])}

            Structured Methods and Results Sections:
            {combined_text}

            Sample References:
            {ref_context}

            INSTRUCTIONS:
            Focus on identifying:
            1. Literature search strategy and databases searched
            2. Key studies included in the review/analysis
            3. Important references that provided data for the synthesis

            Return ONLY valid JSON in this format:
            {{
                "data_sources_description": "Description of the literature search strategy and key studies analyzed",
                "references": [
                    "First author et al., Journal Name, Year",
                    "Second author et al., Journal Name, Year"
                ]
            }}

            JSON:"""
