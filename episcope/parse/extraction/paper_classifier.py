from typing import Dict, List, Tuple, Optional, Union


from pathlib import Path
import json
import re
import logging
import numpy as np
import torch
import ollama
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from collections import defaultdict

from ..blueprints.data_blueprints import StructuredSection, Reference, PaperMetadata, PaperType, ClassificationResult, ClassificationOutput, DataSource, ExtractionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ============================================================================
# PAPER CLASSIFIER
# ============================================================================

class PaperClassifier:
    """Multi-modal paper classification using RAG and semantic similarity."""
    
    # Class-level constants
    TEMPLATE_PARAGRAPHS = {
        "literature_review": [
            "Several studies have examined the relationship between X and Y. Smith et al. (2020) found significant associations, while Jones et al. (2021) reported mixed results. A systematic review by Brown et al. (2019) identified 45 relevant studies.",
            "We conducted a systematic literature search across PubMed, Embase, and Web of Science databases. Studies were included if they met inclusion criteria. Two reviewers independently screened titles and abstracts.",
            "The existing literature can be categorized into three main approaches. Meta-analyses consistently show effect sizes ranging from 0.2 to 0.6. Previous reviews have identified several gaps in the literature.",
            "This scoping review aims to map the available evidence on X. We searched five databases and included 127 studies. The review follows PRISMA guidelines for systematic reviews."
        ],
        "data_analysis": [
            "We analyzed data from the National Health Survey (n=15,432 participants). Data collection occurred between January 2020 and December 2022. Statistical analyses were performed using R version 4.2.",
            "The dataset contained 23,891 observations across 15 variables. Missing data patterns were examined using multiple imputation. Primary outcomes were measured using validated instruments.",
            "Participants were recruited through stratified random sampling. Data were collected through structured interviews. The final analytic sample included 8,734 individuals after exclusions.",
            "Descriptive statistics were calculated for all variables. Logistic regression models were fitted with adjustment for confounders. Effect sizes and 95% confidence intervals are reported."
        ],
        # "methods_tools": [
        #     "We propose a novel statistical method for analyzing longitudinal data with missing values. The method combines multiple imputation with mixed-effects modeling.",
        #     "This paper introduces a new software package for epidemiological analysis. The tool implements advanced causal inference methods and provides user-friendly interfaces.",
        #     "We developed and validated a new questionnaire for measuring health behaviors. Psychometric properties were assessed using factor analysis and reliability testing.",
        #     "The algorithm we present here addresses limitations of existing approaches. Simulation studies demonstrate superior performance under various scenarios."
        # ],
        # "case_study": [
        #     "We report a case series of 15 patients with rare disease X. All patients were treated at our institution between 2018-2023. Clinical characteristics and outcomes are described.",
        #     "This case study examines the implementation of a new intervention in three healthcare settings. We describe barriers, facilitators, and lessons learned.",
        #     "We present findings from a detailed investigation of an outbreak in rural community Y. Contact tracing identified 47 cases over a 6-week period.",
        #     "A 45-year-old patient presented with unusual symptoms. Diagnostic workup revealed a rare condition. This case highlights important clinical considerations."
        # ]
    }
    
    CLASSIFICATION_MAPPING = {
        "A": PaperType.LITERATURE_REVIEW, 
        "B": PaperType.DATA_ANALYSIS,
        # "C": PaperType.METHODS_TOOLS, 
        # "D": PaperType.CASE_STUDY,
        # "E": PaperType.COMMENTARY, 
        # "F": PaperType.OTHER
    }
    
    CATEGORY_LABELS = {
        "A": "Literature Review", 
        "B": "Data Analysis", 
        # "C": "Methods Tools",
        # "D": "Case Study",
        # "E": "Commentary",
        # "F": "Other",
        # "G": "Unclear"
    }
    
    def __init__(self, model_name: str = "deepseek-r1:7b", embedding_model: str = "allenai-specter", use_gpu: bool = True):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Initialize embedding model
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=device)
    
    def get_relevant_chunks(self, metadata: PaperMetadata, index_path: Optional[str] = None, 
                           chunks_path: Optional[str] = None, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Main method to retrieve relevant chunks for each paper type."""
        
        # Initialize result structure
        aggregated_chunks = {paper_type: [] for paper_type in self.TEMPLATE_PARAGRAPHS.keys()}
        query_cache = {}
        
        # Use each template as a query to retrieve relevant chunks
        for paper_type, templates in self.TEMPLATE_PARAGRAPHS.items():
            for template_query in templates:
                template_query = template_query.strip()
                if not template_query:
                    continue
                
                # Cache queries to avoid redundant calls
                if template_query not in query_cache:
                    chunks = self._retrieve_chunks_for_query(template_query, index_path, chunks_path, top_k)
                    query_cache[template_query] = chunks
                
                # Add retrieved chunks to aggregated results
                retrieved_chunks = query_cache[template_query]
                if paper_type in retrieved_chunks:
                    aggregated_chunks[paper_type].extend(retrieved_chunks[paper_type])
        
        # Deduplicate and rank chunks for each paper type
        return self._deduplicate_and_rank_chunks(aggregated_chunks, top_k)
    
    def _retrieve_chunks_for_query(self, query: str, index_path: Optional[str], 
                                  chunks_path: Optional[str], top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve chunks for a single query using FAISS index or fallback to template similarity."""
        
        if not index_path or not chunks_path or not Path(index_path).exists() or not Path(chunks_path).exists():
            return self._fallback_template_similarity(query)
        
        try:
            return self._faiss_search(query, index_path, chunks_path, top_k)
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}, falling back to template similarity")
            return self._fallback_template_similarity(query)
    
    def _faiss_search(self, query: str, index_path: str, chunks_path: str, top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """Perform FAISS-based semantic search."""
        
        # Load FAISS index and chunks
        index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Encode query and search
        query_embedding = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Group results by paper type
        type_chunks = defaultdict(list)
        for score, idx in zip(distances[0], indices[0]):
            if idx < len(chunks_data):
                chunk = chunks_data[idx]
                chunk_text = chunk.get('text', '')
                chunk_type = chunk.get('paper_type', 'unknown')
                
                # Classify unknown chunks
                if chunk_type == 'unknown':
                    chunk_type = self._classify_chunk_by_templates(chunk_text)
                
                type_chunks[chunk_type].append((chunk_text, float(score)))
        
        return dict(type_chunks)
    
    def _fallback_template_similarity(self, query: str) -> Dict[str, List[Tuple[str, float]]]:
        """Fallback method using cosine similarity with template paragraphs."""
        
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        results = {}
        
        for paper_type, templates in self.TEMPLATE_PARAGRAPHS.items():
            template_embeddings = self.embedder.encode(templates, show_progress_bar=False)
            similarities = cosine_similarity(query_embedding, template_embeddings)
            
            # Create results with similarity scores
            type_results = [(template, float(similarities[0][i])) for i, template in enumerate(templates)]
            results[paper_type] = sorted(type_results, key=lambda x: x[1], reverse=True)
        
        return results
    
    def _classify_chunk_by_templates(self, chunk_text: str) -> str:
        """Classify a chunk based on template similarity."""
        
        chunk_embedding = self.embedder.encode([chunk_text], show_progress_bar=False)
        best_type, best_score = "other", -1
        
        for paper_type, templates in self.TEMPLATE_PARAGRAPHS.items():
            template_embeddings = self.embedder.encode(templates, show_progress_bar=False)
            similarities = cosine_similarity(chunk_embedding, template_embeddings)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_type = paper_type
        
        return best_type
    
    def _deduplicate_and_rank_chunks(self, aggregated_chunks: Dict[str, List[Tuple[str, float]]], 
                                    top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """Remove duplicates and keep top-k chunks for each paper type."""
        
        final_chunks = {}
        for paper_type, chunks in aggregated_chunks.items():
            # Deduplicate by keeping highest score for each unique text
            unique_chunks = {}
            for text, score in chunks:
                text = text.strip()
                if text and (text not in unique_chunks or score > unique_chunks[text]):
                    unique_chunks[text] = score
            
            # Sort by score and keep top-k
            sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_chunks[paper_type] = [(text, score) for text, score in sorted_chunks]
        
        return final_chunks
    
    def classify_based_on_relevant_chunks(self, relevant_chunks: Dict[str, List[Tuple[str, float]]], 
                                         metadata: PaperMetadata) -> ClassificationResult:
        """Classify paper based on relevant chunks using LLM."""
        
        # Check if we have any chunks
        if not any(chunks for chunks in relevant_chunks.values()):
            logger.warning("No relevant chunks found, using template fallback")
            all_templates = [template for templates in self.TEMPLATE_PARAGRAPHS.values() for template in templates]
            relevant_chunks = self._fallback_template_similarity(" ".join(all_templates))
        
        return self._llm_classify(metadata, relevant_chunks)
    
    def _llm_classify(self, metadata: PaperMetadata, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> ClassificationResult:
        """Perform LLM-based classification."""
        
        # Build chunks information
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        
        # Create structured prompt
        prompt = self._create_classification_prompt(metadata, chunks_info)
        
        # Get LLM response with retry logic
        for attempt in range(2):
            try:
                response = self._call_llm_with_schema(prompt)
                return self._parse_classification_response(response)
            except Exception as e:
                logger.warning(f"Classification attempt {attempt + 1} failed: {e}")
                if attempt == 1:  # Last attempt
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
    

                # C – Methods Development (new algorithms or software)
                # D – Descriptive Study (case report, implementation study)
                # E – Editorial/Opinion
                # F – Other

    def _create_classification_prompt(self, metadata: PaperMetadata, chunks_info: str) -> str:
        """Create the classification prompt."""
        
        schema = ClassificationOutput.model_json_schema()
        
        return f"""You are an expert academic classifier. Your task is to determine the primary type of research a paper represents by classifying it into one of the following categories, A to F.

                **Categories:**
                A – Literature Review (systematic review, meta-analysis)
                B – Empirical Study (uses original data)


                **Paper Content:**
                Title: {metadata.title}
                Abstract: {metadata.abstract or 'N/A'}
                Keywords: {', '.join(metadata.keywords or [])}

                **Relevant Extracts:**
                (These extracts are provided to give you additional context, grouped by semantic similarity. You must consider all of them to identify the paper's core contribution.):
                {chunks_info}

                **Instructions:**
                1. **Analyze the evidence:** Review all the provided information to determine what the paper's main contribution is. For instance, does it primarily synthesize existing studies, or does it analyze new, original data?
                2. **Classify:** Select a single letter from A to F that best represents the paper's primary classification.
                3. **Format the output:** Return only a single JSON object that strictly adheres to the provided schema. Do not add any extra text or commentary.

                **Schema:**
{json.dumps(schema)}
"""
    
    def _call_llm_with_schema(self, prompt: str) -> str:
        """Call LLM with structured output."""
        
        schema = ClassificationOutput.model_json_schema()
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0},
            format=schema
        )
        logging.info(f"Classification response: {response.message.content}")
        # Extract content from response
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = getattr(response, "message", None)
            if content:
                content = getattr(content, "content", "")
        
        # Remove thinking tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    
    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        parsed = ClassificationOutput.model_validate_json(response_content)
        
        # Map classification letter to paper type
        paper_type = self.CLASSIFICATION_MAPPING.get(parsed.classification, PaperType.DATA_ANALYSIS) # PaperType.OTHER)
        
        # Build class probabilities
        class_probs = parsed.class_probabilities or {}
        if parsed.classification in self.CATEGORY_LABELS:
            class_probs[self.CATEGORY_LABELS[parsed.classification]] = parsed.confidence or 0.9
        
        # Fill missing probabilities
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
            paper_type=PaperType.DATA_ANALYSIS, # OTHER
            confidence=0.,
            class_probabilities={label: 0.16 for label in self.CATEGORY_LABELS.values()},
            evidence={"reasoning": "Classification failed, defaulting to unclear"}
        )


# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class DataExtractor:
    """Enhanced LLM extractor using structured GROBID data with integrated classification."""
    
    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        self.client = ollama.Client()
        self.classifier = PaperClassifier(model_name)
    
    def analyze_and_extract(self, sections: List[StructuredSection], metadata: PaperMetadata,
                           references: List[Reference], relevant_chunks: List[Dict], 
                           query: str) -> Tuple[PaperType, ExtractionResult, Optional[Dict]]:
        """Complete pipeline: classify paper type and extract relevant information."""
        
        # Step 1: Get relevant chunks for classification
        logger.info("Retrieving relevant chunks...")
        chunk_dict = self.classifier.get_relevant_chunks(metadata)
        
        # Step 2: Classify the paper
        logger.info("Classifying paper type...")
        classification_result = self.classifier.classify_based_on_relevant_chunks(chunk_dict, metadata)
        paper_type = classification_result.paper_type
        
        logger.info(f"Paper classified as: {paper_type.value} (confidence: {classification_result.confidence:.2f})")
        
        # Step 3: Extract data sources based on paper type
        extraction_result, confidence_scores = self._extract_by_paper_type(
            paper_type, relevant_chunks, references, metadata, query, sections
        )
        
        # Add classification info to confidence scores
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
        
        extraction_methods = {
            PaperType.DATA_ANALYSIS: self._extract_direct_sources,
            PaperType.LITERATURE_REVIEW: self._extract_literature_sources,
            # PaperType.OTHER: self._extract_other_sources,
            # PaperType.UNCLEAR: self._extract_mixed_sources
        }
        
        method = extraction_methods.get(paper_type, self._extract_other_sources)
        logger.info(f"Using extraction method for {paper_type.value}")
        
        return method(chunks, references, metadata, query, sections)
    
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
    
    def _extract_other_sources(self, chunks: List[Dict], references: List[Reference], metadata: PaperMetadata, 
                              query: str, sections: List[StructuredSection]) -> Tuple[ExtractionResult, Dict]:
        """Extract sources for other types of papers."""
        
        relevant_text = self._get_section_text(sections, ['Methods', 'Implementation', 'Case Study', 'Examples'])
        chunk_text = self._compile_chunk_text(chunks, ['Methods', 'Results'])
        combined_text = f"{relevant_text}\n\n=== RELEVANT EXCERPTS ===\n{chunk_text}"
        
        prompt = self._create_other_sources_prompt(metadata, combined_text)
        
        return self._execute_extraction(prompt, chunks, is_references=False)
    
    def _extract_mixed_sources(self, chunks: List[Dict], references: List[Reference], metadata: PaperMetadata, 
                              query: str, sections: List[StructuredSection]) -> Tuple[ExtractionResult, Dict]:
        """Extract sources for unclear paper types using both approaches."""
        
        logger.info("Paper type unclear, trying both direct and literature approaches")
        
        # Try both approaches
        direct_result, direct_conf = self._extract_direct_sources(chunks, references, metadata, query, sections)
        lit_result, lit_conf = self._extract_literature_sources(chunks, references, metadata, query, sections)
        
        # Combine results
        combined_result = ExtractionResult(
            description=f"Mixed extraction (unclear paper type): {direct_result.description} | {lit_result.description}",
            data_sources=direct_result.data_sources,
            references=lit_result.references
        )
        
        return combined_result, direct_conf or {}
    
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
    
    def _create_other_sources_prompt(self, metadata: PaperMetadata, combined_text: str) -> str:
        """Create prompt for other source types extraction."""
        
        return f"""Extract relevant sources from this research paper (methodological/theoretical/other type).

            Paper Title: {metadata.title}
            Abstract: {metadata.abstract or 'Not available'}

            Relevant Sections:
            {combined_text}

            INSTRUCTIONS:
            This paper doesn't fit the typical data analysis or literature review categories.
            Focus on identifying any:
            1. Datasets used for examples or validation
            2. Key references that inform the methodology
            3. Sources that provide context or background

            Return ONLY valid JSON in this format:
            {{
                "data_sources_description": "Description of sources and references relevant to this work",
                "data_sources": [],
                "references": []
            }}

            JSON:"""
    
    def _execute_extraction(self, prompt: str, chunks: List[Dict], is_references: bool = False) -> Tuple[ExtractionResult, Dict]:
        """Execute the extraction with error handling."""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={'temperature': 0}
            )
            
            response_text = response['message']['content'].strip()
            
            # Clean response
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            return self._parse_extraction_response(response_text, is_references)
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult("Extraction failed"), {}
    
    def _parse_extraction_response(self, response: str, is_references: bool = False) -> Tuple[ExtractionResult, Dict]:
        """Parse JSON response from extraction LLM."""
        
        try:
            # Find and clean JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in extraction response")
                return ExtractionResult("No JSON found"), {}
            
            json_str = json_match.group(0)
            # Clean up common JSON formatting issues
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            data = json.loads(json_str)
            description = data.get('data_sources_description', 'No description provided')
            
            # Parse data sources
            data_sources = []
            for ds_data in data.get('data_sources', []):
                if isinstance(ds_data, dict):
                    data_sources.append(DataSource(
                        source_name=ds_data.get('source_name', ''),
                        url=ds_data.get('url', 'N/A'),
                        explanation=ds_data.get('explanation', ''),
                        section_found=ds_data.get('section_found', '')
                    ))
            
            # Parse references
            references = []
            for ref_text in data.get('references', []):
                if isinstance(ref_text, str) and ref_text.strip():
                    references.append(Reference(raw_text=ref_text.strip()))
            
            # Create extraction result
            result = ExtractionResult(
                description=description,
                data_sources=data_sources,
                references=references
            )
            
            # Create confidence scores (placeholder for future enhancement)
            confidence_scores = {
                "extraction_method": "llm_based",
                "json_parsing_success": True,
                "num_data_sources": len(data_sources),
                "num_references": len(references)
            }
            
            return result, confidence_scores
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return ExtractionResult(f"JSON parsing failed: {str(e)}"), {"json_parsing_success": False}
        except Exception as e:
            logger.error(f"Extraction parsing failed: {e}")
            return ExtractionResult(f"Parsing failed: {str(e)}"), {"parsing_success": False}


# ============================================================================
# UTILITY FUNCTIONS AND CONSTANTS
# ============================================================================

def create_paper_system(model_name: str = "deepseek-r1:7b", 
                        embedding_model: str = "all-MiniLM-L6-v2",
                        use_gpu: bool = True) -> Tuple[PaperClassifier, DataExtractor]:
    """
    Factory function to create classifier and extractor instances.
    
    Args:
        model_name: Name of the Ollama model
        embedding_model: Name of the sentence transformer model
        use_gpu: Whether to use GPU for embeddings
        
    Returns:
        Tuple of (classifier, extractor) instances
    """
    classifier = PaperClassifier(model_name, embedding_model, use_gpu)
    extractor = DataExtractor(model_name)
    return classifier, extractor


def validate_paper_metadata(metadata: PaperMetadata) -> bool:
    """
    Validate that paper metadata contains minimum required information.
    
    Args:
        metadata: Paper metadata to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not metadata.title or not metadata.title.strip():
        logger.warning("Paper metadata missing title")
        return False
    
    # At least title is required, abstract is highly recommended
    if not metadata.abstract:
        logger.warning("Paper metadata missing abstract - classification may be less accurate")
    
    return True


def format_classification_summary(result: ClassificationResult) -> str:
    """
    Format classification result into human-readable summary.
    
    Args:
        result: Classification result to format
        
    Returns:
        Formatted summary string
    """
    summary = f"Paper Type: {result.paper_type.value.replace('_', ' ').title()}\n"
    summary += f"Confidence: {result.confidence:.2%}\n"
    
    if result.class_probabilities:
        summary += "\nClass Probabilities:\n"
        sorted_probs = sorted(result.class_probabilities.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs[:3]:  # Show top 3
            summary += f"  {class_name}: {prob:.2%}\n"
    
    if "reasoning" in result.evidence:
        summary += f"\nReasoning: {result.evidence['reasoning']}\n"
    
    return summary


def format_extraction_summary(result: ExtractionResult) -> str:
    """
    Format extraction result into human-readable summary.
    
    Args:
        result: Extraction result to format
        
    Returns:
        Formatted summary string
    """
    summary = f"Description: {result.description}\n\n"
    
    if result.data_sources:
        summary += f"Data Sources ({len(result.data_sources)}):\n"
        for i, ds in enumerate(result.data_sources, 1):
            summary += f"  {i}. {ds.source_name}\n"
            if ds.explanation:
                summary += f"     {ds.explanation}\n"
            if ds.url != "N/A":
                summary += f"     URL: {ds.url}\n"
            summary += "\n"
    
    if result.references:
        summary += f"Key References ({len(result.references)}):\n"
        for i, ref in enumerate(result.references[:10], 1):  # Limit to first 10
            summary += f"  {i}. {ref.raw_text}\n"
    
    return summary

