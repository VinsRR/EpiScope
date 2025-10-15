import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity

from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.vectordb.base import AbstractVectorDB
from episcope.utils.data_blueprints import (
    ClassificationOutput,
    ClassificationResult,
    PaperMetadata,
    PaperType,
    ExtractionResult,
    DataSource,
    Reference,
    StructuredSection,
)
from episcope.parse_configs import PaperClassifierConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperClassifier:
    """Multi-modal paper classification using RAG and semantic similarity."""

    TEMPLATE_PARAGRAPHS = {
        "literature_review": [
            "Several studies have examined the relationship between X and Y. Smith et al. (2020) found significant associations, while Jones et al. (2021) reported mixed results. A systematic review by Brown et al. (2019) identified 45 relevant studies.",
            "We conducted a systematic literature search across PubMed, Embase, and Web of Science databases. Studies were included if they met inclusion criteria. Two reviewers independently screened titles and abstracts.",
        ],
        "data_analysis": [
            "We analyzed data from the National Health Survey (n=15,432 participants). Data collection occurred between January 2020 and December 2022. Statistical analyses were performed using R version 4.2.",
            "The dataset contained 23,891 observations across 15 variables. Missing data patterns were examined using multiple imputation. Primary outcomes were measured using validated instruments.",
        ],
    }
    CLASSIFICATION_MAPPING = {"A": PaperType.LITERATURE_REVIEW, "B": PaperType.DATA_ANALYSIS}
    CATEGORY_LABELS = {"A": "Literature Review", "B": "Data Analysis"}

    def __init__(self, model_name: str, embedder: SimplifiedEmbedder, vectordb: AbstractVectorDB,
                 config: Optional[PaperClassifierConfig] = None):
        self.model_name = model_name
        self.client = ollama.Client()
        self.embedder = embedder
        self.vectordb = vectordb
        self.config = config or PaperClassifierConfig()

    def get_relevant_chunks(self, metadata: PaperMetadata, paper_id: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Main method to retrieve relevant chunks for each paper type."""
        aggregated_chunks = defaultdict(list)
        for paper_type, templates in self.TEMPLATE_PARAGRAPHS.items():
            for template_query in templates:
                retrieved_chunks = self._retrieve_chunks_for_query(template_query, paper_id, top_k)
                for chunk in retrieved_chunks:
                    # Simple classification of retrieved chunk to the query's paper type
                    aggregated_chunks[paper_type].append((chunk.get("text", ""), chunk.get("score", 0.0)))
        
        return self._deduplicate_and_rank_chunks(aggregated_chunks, top_k)

    def _retrieve_chunks_for_query(self, query: str, paper_id: str, top_k: int) -> List[Dict]:
        """Retrieve chunks for a single query using the vector DB."""
        if not paper_id:
            return []
        try:
            query_embedding = self.embedder.embed_text(query)
            results = self.vectordb.search(query_embedding, top_k=top_k, namespace=paper_id)
            return results
        except Exception as e:
            logger.warning(f"VectorDB search failed for paper {paper_id}: {e}")
            return []

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

    def _llm_classify(self, metadata: PaperMetadata, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> ClassificationResult:
        """Perform LLM-based classification."""
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        prompt = self._create_classification_prompt(metadata, chunks_info)
        
        for attempt in range(2):
            try:
                response = self._call_llm_with_schema(prompt)
                return self._parse_classification_response(response)
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

    def _create_classification_prompt(self, metadata: PaperMetadata, chunks_info: str) -> str:
        """Create the classification prompt."""
        schema = ClassificationOutput.model_json_schema()
        return f"""You are an expert academic classifier. Your task is to determine the primary type of a research paper.
                **Categories:**
                A – Literature Review (systematic review, meta-analysis)
                B – Empirical Study (uses original data)

                **Paper Content:**
                Title: {metadata.title}
                Abstract: {metadata.abstract or 'N/A'}
                Keywords: {', '.join(metadata.keywords or [])}

                **Relevant Extracts:**
                {chunks_info}

                **Instructions:**
                1. Analyze the evidence to determine the paper's main contribution.
                2. Select a single letter (A or B) that best represents the paper's primary classification.
                3. Return a single JSON object adhering to the schema. Do not add extra text.

                **Schema:**
{json.dumps(schema)}
"""

    def _call_llm_with_schema(self, prompt: str) -> str:
        """Call LLM with structured output."""
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0},
            format="json"
        )
        content = response.get("message", {}).get("content", "")
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        parsed = ClassificationOutput.model_validate_json(response_content)
        paper_type = self.CLASSIFICATION_MAPPING.get(parsed.classification, PaperType.DATA_ANALYSIS)
        
        class_probs = parsed.class_probabilities or {}
        if parsed.classification in self.CATEGORY_LABELS:
            class_probs[self.CATEGORY_LABELS[parsed.classification]] = parsed.confidence or 0.9
        
        for label in self.CATEGORY_LABELS.values():
            if label not in class_probs:
                class_probs[label] = 0.01
        
        confidence = parsed.confidence or class_probs.get(self.CATEGORY_LABELS.get(parsed.classification, "Other"), 0.9)
        
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
            class_probabilities={label: 0.5 for label in self.CATEGORY_LABELS.values()},
            evidence={"reasoning": "Classification failed, defaulting to data analysis"}
        )

class DataExtractor:
    """Enhanced LLM extractor using structured GROBID data with integrated classification."""
    
    def __init__(self, model_name: str, embedder: SimplifiedEmbedder, vectordb: AbstractVectorDB):
        self.model_name = model_name
        self.client = ollama.Client()
        self.classifier = PaperClassifier(model_name, embedder, vectordb)
    
    def analyze_and_extract(self, sections: List[StructuredSection], metadata: PaperMetadata,
                           references: List[Reference], relevant_chunks: List[Dict], 
                           query: str, paper_id: str) -> Tuple[PaperType, ExtractionResult, Optional[Dict]]:
        """Complete pipeline: classify paper type and extract relevant information."""
        
        logger.info("Retrieving relevant chunks for classification...")
        chunk_dict = self.classifier.get_relevant_chunks(metadata, paper_id)
        
        logger.info("Classifying paper type...")
        classification_result = self.classifier.classify_based_on_relevant_chunks(chunk_dict, metadata)
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
                              metadata: PaperMetadata, query: str, sections: List[StructuredSection]) -> Tuple[ExtractionResult, Dict]:
        """Route extraction based on paper type."""
        method = {
            PaperType.DATA_ANALYSIS: self._extract_direct_sources,
            PaperType.LITERATURE_REVIEW: self._extract_literature_sources,
        }.get(paper_type, self._extract_direct_sources)
        
        logger.info(f"Using extraction method for {paper_type.value}")
        return method(chunks, references, metadata, query, sections)

    def _execute_extraction(self, prompt: str, chunks: List[Dict], is_references: bool = False) -> Tuple[ExtractionResult, Dict]:
        """Execute the extraction with error handling."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={'temperature': 0},
                format="json"
            )
            response_text = response['message']['content'].strip()
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
            
            data_sources = [DataSource(**ds_data) for ds_data in data.get('data_sources', []) if isinstance(ds_data, dict)]
            references = [Reference(raw_text=ref_text.strip()) for ref_text in data.get('references', []) if isinstance(ref_text, str)]
            
            result = ExtractionResult(description=description, data_sources=data_sources, references=references)
            confidence_scores = {"extraction_method": "llm_based", "json_parsing_success": True, "num_data_sources": len(data_sources), "num_references": len(references)}
            
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