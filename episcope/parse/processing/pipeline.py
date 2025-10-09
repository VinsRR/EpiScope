# pdfs = list(Path(input_dir).rglob("*.pdf"))
# if max_files:
#     pdfs = [p for p in pdfs if GT_PAPER_TYPES.get(p.stem, "") == SELECTED_TYPE]
#     # print([p.stem for p in pdfs])
#     pdfs = pdfs[:max_files]
#     logger.info(f"Limiting to {len(pdfs)} files")



import os
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..blueprints.data_blueprints import StructuredSection, Reference, PaperMetadata
from ..configs.configs import GT_PAPER_TYPES, SELECTED_TYPE, SearchConfig, PipelineConfig
from ..indexing.embeddings import EmbeddingIndexer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# -------------------------
# Search Strategy Classes
# -------------------------
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    """Represents a search result chunk with metadata."""
    id: str
    text: str
    section_type: str = "other"
    title: str = ""
    similarity_score: float = 0.0
    source: str = ""  # semantic, keyword, context
    artifacts: Dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0




class TextProcessor:
    """Handles text normalization and pattern matching."""
    
    URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>{}\[\]]+\b", re.IGNORECASE)
    DOI_URL_PATTERN = re.compile(r"(?:https?://(?:doi\.org|dx\.doi\.org)/\s*)?(10\.\d{4,9}/[^\s\"'<>{}\[\]]+)", re.IGNORECASE)
    ACCESSION_PATTERN = re.compile(r"\b(?:GSE|PRJNA|SRP|ERP|SRR|SRA|ENA|PRJEB)\d{3,7}\b", re.IGNORECASE)
    SENTENCE_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
    
    AVAILABILITY_TERMS = {
        "data available", "data are available", "data is available", "available at", 
        "available from", "available upon request", "upon request", "available on request",
        "data availability", "accession number", "deposited in", "deposited at",
        "uploaded to", "figshare", "zenodo", "github", "dryad", "ncbi", "sra", "ena"
    }
    
    REPOSITORIES = {"zenodo", "figshare", "dryad", "github", "ncbi", "gisaid", "sra", "ena", "dataverse"}
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize text for robust pattern matching."""
        if not text:
            return ""
        
        # Remove soft hyphens and normalize spaces
        text = text.replace("\u00AD", "").replace("\u00A0", " ")
        
        # Fix common broken patterns
        text = re.sub(r"https?\s*:\s*/\s*/", "https://", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdoi\s*[.:]?\s*org\s*/\s*", "doi.org/", text, flags=re.IGNORECASE)
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"/\s+", "/", text)
        text = re.sub(r"\.\s+", ".", text)
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    @classmethod
    def extract_artifacts(cls, text: str) -> Dict[str, Any]:
        """Extract URLs, DOIs, and accession numbers from text."""
        normalized = cls.normalize_text(text)
        
        urls = cls.URL_PATTERN.findall(normalized)
        
        # Extract DOIs from both patterns
        dois = set()
        for match in cls.DOI_PATTERN.finditer(normalized):
            dois.add(match.group())
        for match in cls.DOI_URL_PATTERN.finditer(normalized):
            dois.add(match.group(1))
        
        accessions = cls.ACCESSION_PATTERN.findall(normalized)
        
        availability_score = sum(1 for term in cls.AVAILABILITY_TERMS 
                               if term in normalized.lower())
        
        return {
            "urls": urls,
            "dois": list(dois),
            "accessions": accessions,
            "availability_score": availability_score,
            "has_artifacts": bool(urls or dois or accessions),
            "normalized_text": normalized
        }


class QueryGenerator:
    """Generates queries for different paper types."""
    
    PAPER_TYPE_QUERIES = {
        "data_analysis": {
            "primary": [
                'Supplementary data are available at"', 
                '"Detailed data of this study are provided in the Supplementary Information and will be available upon"',
                '"source data is available at"',
                "The data collection integrates multiple data sources"
            ],
            "secondary": [
                "Based on the data provided by",
                "We collected data on",
                
            ]
        },

        "literature_review": {
            "primary": [
                'The full list of included studies is available at', 
                'Appendix: List of included studies', 
                'The list of data entries for all included studies are provided in',
                'Supplementary material for this article can be found online at'
            ],
            "secondary": [
            ]
        },
        "methods_tools": {
            "primary": [
                '"code available"', '"repository"', '"github"',
                '"software availability"'
            ],
            "secondary": [
                "implementation benchmark dataset",
                "source code docker package"
            ]
        },
        "case_study": {
            "primary": [
                '"data availability"', "surveillance data",
                "institutional data"
            ],
            "secondary": [
                "case study dataset",
                "outbreak data collection"
            ]
        }
    }
    
    @classmethod
    def generate_queries(cls, paper_type: str, metadata=None, hyde_callable=None) -> List[Tuple[str, float, str]]:
        """Generate weighted queries for the given paper type."""
        queries = []
        
        # Get paper-type specific queries
        type_queries = cls.PAPER_TYPE_QUERIES.get(paper_type, cls.PAPER_TYPE_QUERIES["data_analysis"])
        
        # Add primary queries (high weight)
        for query in type_queries["primary"]:
            queries.append((query, 5.0, f"primary_{paper_type}"))
        
        # Add secondary queries (medium weight)
        for query in type_queries["secondary"]:
            queries.append((query, 3.0, f"secondary_{paper_type}"))
        
        # # Add repository-specific queries
        # for repo in TextProcessor.REPOSITORIES:
        #     queries.append((f'"{repo}" data', 3.5, f"repo_{repo}"))
        
        # # Add title-aware query if available
        # if metadata and hasattr(metadata, 'title') and metadata.title:
        #     title_query = f'data availability "{metadata.title}"'
        #     queries.append((title_query, 4.0, "title_aware"))
        
        # Generate HyDE queries if available
        if hyde_callable and metadata:
            try:
                hyde_queries = hyde_callable(f"Generate data availability statements for {paper_type} paper")
                for i, hq in enumerate(hyde_queries[:3]):
                    queries.append((hq, 3.5, f"hyde_{i}"))
            except Exception as e:
                logger.debug(f"HyDE generation failed: {e}")
        
        # Deduplicate and sort by weight
        unique_queries = {}
        for query, weight, reason in queries:
            key = query.lower().strip()
            if key not in unique_queries or weight > unique_queries[key][1]:
                unique_queries[key] = (query, weight, reason)
        
        result = list(unique_queries.values())
        result.sort(key=lambda x: x[1], reverse=True)
        return result



class ChunkSearcher:
    """Handles different search strategies for finding relevant chunks."""
    
    def __init__(self, config: SearchConfig, querier, text_processor: TextProcessor):
        self.config = config
        self.querier = querier
        self.text_processor = text_processor
    
    def search_semantic(self, queries: List[Tuple[str, float, str]], 
                       index_path: str, chunks_path: str) -> List[SearchResult]:
        """Perform semantic search using weighted queries."""
        if not self.config.use_semantic_search:
            return []
        
        results = []
        seen_ids = set()
        
        for query, weight, reason in queries:
            try:
                chunks = self.querier.query_structured_chunks(
                    query=query,
                    index_path=index_path,
                    chunks_path=chunks_path,
                    top_k=self.config.top_k_semantic,
                    similarity_threshold=self.config.similarity_threshold
                )
                
                for chunk in chunks:
                    chunk_id = str(chunk.get("id", ""))
                    if chunk_id in seen_ids:
                        continue
                    
                    artifacts = self.text_processor.extract_artifacts(chunk.get("text", ""))
                    
                    result = SearchResult(
                        id=chunk_id,
                        text=chunk.get("text", ""),
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=chunk.get("similarity_score", 0.0),
                        source=f"semantic_{reason}",
                        artifacts=artifacts
                    )
                    
                    results.append(result)
                    seen_ids.add(chunk_id)
                    
            except Exception as e:
                logger.debug(f"Semantic search failed for query '{query}': {e}")
        
        return results
    
    def search_keyword(self, chunks_path: str) -> List[SearchResult]:
        """Perform keyword-based search."""
        if not self.config.use_keyword_search:
            return []
        
        keywords = ["data availability", "available", "repository", "accession", 
                   "deposited", "supplementary", "github", "zenodo"]
        
        try:
            chunks = self.querier.query_keyword_chunks(
                chunks_path=chunks_path,
                keywords=keywords,
                top_k=self.config.top_k_keyword
            )
            
            results = []
            for chunk in chunks:
                artifacts = self.text_processor.extract_artifacts(chunk.get("text", ""))
                
                result = SearchResult(
                    id=str(chunk.get("id", "")),
                    text=chunk.get("text", ""),
                    section_type=chunk.get("section_type", "other"),
                    title=chunk.get("title", ""),
                    similarity_score=0.0,
                    source="keyword",
                    artifacts=artifacts
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.debug(f"Keyword search failed: {e}")
            return []
    
    def extract_context_snippets(self, chunks_path: str) -> List[SearchResult]:
        """Extract context snippets around availability terms."""
        if not self.config.use_context_extraction:
            return []
        
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read chunks for context extraction: {e}")
            return []
        
        results = []
        snippet_id = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("content", ""))
            if not text:
                continue
            
            normalized = self.text_processor.normalize_text(text)
            sentences = self.text_processor.SENTENCE_SPLIT.split(normalized)
            
            for sent_idx, sentence in enumerate(sentences):
                artifacts = self.text_processor.extract_artifacts(sentence)
                
                # Only keep sentences with availability indicators or artifacts
                if artifacts["availability_score"] > 0 or artifacts["has_artifacts"]:
                    # Create context window (sentence before and after)
                    context_sentences = []
                    if sent_idx > 0:
                        context_sentences.append(sentences[sent_idx - 1])
                    context_sentences.append(sentence)
                    if sent_idx < len(sentences) - 1:
                        context_sentences.append(sentences[sent_idx + 1])
                    
                    context_text = " ".join(context_sentences).strip()
                    
                    result = SearchResult(
                        id=f"context_{snippet_id}",
                        text=context_text,
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=0.0,
                        source="context",
                        artifacts=artifacts
                    )
                    
                    results.append(result)
                    snippet_id += 1
        
        return results

class ResultRanker: # RERANKER MIGHT BE POWERFUL IF TRAINED
    """Ranks and deduplicates search results."""
    
    @staticmethod
    def calculate_rank_score(result: SearchResult, query_weight: float = 1.0) -> float:
        """Calculate ranking score for a search result."""
        base_score = result.similarity_score * query_weight
        
        # Boost for artifacts
        artifact_bonus = 0.0
        if result.artifacts.get("urls"):
            artifact_bonus += 2.0
        if result.artifacts.get("dois"):
            artifact_bonus += 1.8
        if result.artifacts.get("accessions"):
            artifact_bonus += 1.5
        
        # Boost for availability terms
        availability_bonus = result.artifacts.get("availability_score", 0) * 0.5
        
        return base_score + artifact_bonus + availability_bonus
    
    @staticmethod
    def deduplicate_and_rank(results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Deduplicate by text similarity and rank by score."""
        if not results:
            return []
        
        # Calculate rank scores
        for result in results:
            result.rank_score = ResultRanker.calculate_rank_score(result)
        
        # Simple deduplication by normalized text signature
        unique_results = {}
        for result in results:
            # Create signature from first 200 chars of normalized text
            signature = result.artifacts.get("normalized_text", result.text)[:200].lower().strip()
            
            if signature not in unique_results or result.rank_score > unique_results[signature].rank_score:
                unique_results[signature] = result
        
        # Sort by rank score and return top-k
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.rank_score, reverse=True)
        
        return final_results[:top_k]

# -------------------------
# Main Pipeline Processor
# -------------------------
class PipelineProcessor:
    """Main pipeline processor with simplified architecture."""
    
    def __init__(self, output_dir: str = "output", index_dir: str = "indices", 
                 config: Optional[PipelineConfig] = None):
        self.output_dir = Path(output_dir)
        self.index_dir = Path(index_dir)
        self.config = config or PipelineConfig()
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self._init_components()
        
        self.results = []
        self.results_file = self.output_dir / "_extraction_results.csv"
    
    def _init_components(self):
        """Initialize all pipeline components."""
        from ..extraction.paper_classifier import PaperClassifier
        from ..querying.rag_querier import RAGQuerier
        from ..extraction.llm_extractor import LLMExtractor
        from ...retrieve.embeddings import SimplifiedEmbedder

        # Create a single embedder instance for consistency
        embedder = SimplifiedEmbedder(embed_model=self.config.embedding_model)

        # Initialize components with the shared embedder
        self.indexer = EmbeddingIndexer(model_name=self.config.embedding_model)
        self.paper_classifier = PaperClassifier(self.config.classifier_model, embedder=embedder)
        self.querier = RAGQuerier(embedder=embedder)
        self.extractor = LLMExtractor(self.config.llm_model)
        
        # Initialize text processor and searcher
        self.text_processor = TextProcessor()
        self.chunk_searcher = ChunkSearcher(self.config.search, self.querier, self.text_processor)
        
        # Initialize HyDE if configured
        self.hyde_generate = None
        if self.config.use_hyde:
            try:
                from ..querying.hyde import HYDE
                hyde = HYDE(self.config.hyde_model)
                self.hyde_generate = lambda prompt: list(hyde.generate(prompt, n=4))
            except Exception:
                logger.debug("HyDE not available")
    
    def process_pdf(self, pdf_path: str, storage_path: str = None, strategy_name: str = "default") -> Dict[str, Any]:
        """Process a single PDF file."""
        paper_id = Path(pdf_path).stem
        base_dir = Path(pdf_path).parent if storage_path is None else Path(storage_path)
        
        logger.info(f"Processing {paper_id}")
        
        # Load extracted data
        metadata, sections, references = self._load_extracted_data(base_dir, paper_id)
        
        # Create/load index and chunks
        index_path, chunks_path = self._prepare_index_and_chunks(sections, metadata, paper_id, base_dir, strategy_name)
        
        # Classify paper type
        paper_type = self._classify_paper_type(metadata, index_path, chunks_path)
        
        # Extract data sources
        extraction_result, confidence_scores = self._extract_data_sources(
            paper_type, index_path, chunks_path, references, pdf_path, metadata
        )
        
        # Create result dictionary
        result_dict = self._create_result_dict(
            paper_id, pdf_path, metadata, sections, references, 
            paper_type, extraction_result, confidence_scores
        )

        result_markdown = self._create_result_markdown(paper_id, pdf_path, metadata, sections, references,
            paper_type, extraction_result, confidence_scores)

        logger.info(f"Completed processing {paper_id}")
        return result_dict, result_markdown
    
    def _load_extracted_data(self, base_dir: Path, paper_id: str):
        """Load metadata, sections, and references from JSON files."""
        try:
            from ..blueprints.data_blueprints import PaperMetadata, SectionList, ReferenceList
            # sections = SectionList.from_json(str(base_dir / f"{paper_id}_sections.json")).items
            metadata = PaperMetadata.from_json(Path(base_dir, paper_id,"metadata.json")) 
            sections = SectionList.from_json(Path(base_dir, paper_id,"sections.json")) 
            references = ReferenceList.from_json(Path(base_dir, paper_id,"references.json"))

            return metadata, sections, references
            
        except Exception as e:
            logger.error(f"Failed to load extracted data for {paper_id}: {e}")
            raise
    
    def _prepare_index_and_chunks(self, sections, metadata, paper_id: str, base_dir: Path, strategy_name: str):
        """Prepare index and chunks files."""
        # Create index if indexer is available
        index_path = None
        if self.indexer:
            try:
                index_path = self.indexer.create_index(sections, metadata, str(self.index_dir), paper_id)
            except Exception as e:
                logger.warning(f"Index creation failed for {paper_id}: {e}")
        
        # Ensure chunks file exists, accounting for strategy in directory structure
        strategy_dir = self.index_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True, parents=True)
        chunks_path = strategy_dir / f"{paper_id}_structured_chunks.json"

        if not chunks_path.exists():
            chunks = []
            for i, section in enumerate(sections):
                if isinstance(section, StructuredSection):
                    # Access for StructuredSection instances
                    chunk = {
                        "id": i,
                        "section_type": section.section_type,
                        "title": section.title,
                        "text": section.content,
                        "strategy": strategy_name,
                    }
                elif isinstance(section, dict):
                    chunk = {
                        "id": i,
                        "section_type": section.get("section_type", "other"),
                        "title": section.get("title", ""),
                        "text": section.get("content", ""),
                        "strategy": strategy_name,
                    }
                else:
                    # Handle unexpected types gracefully
                    assert False, f"Unexpected section type: {type(section)}"

                chunks.append(chunk)
                
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        return str(index_path) if index_path else "", str(chunks_path)
    
    def _classify_paper_type(self, metadata, index_path: str, chunks_path: str) -> str:
        """Classify the paper type."""
        try:
            # go = False
            # if go:
                # index_path = index_path or str(self.index_dir / f"{paper_id}.index")
            relevant_chunks = self.paper_classifier.get_relevant_chunks(metadata=metadata, index_path=str(index_path), chunks_path=str(chunks_path))
            classification_result = self.paper_classifier.classify_based_on_relevant_chunks(relevant_chunks, metadata)
            paper_type = classification_result.paper_type.value
            # else:
            #     paper_type = "data_analysis" if SELECTED_TYPE == "data" else "literature_review"
        except Exception as e:
            logger.warning(f"Paper classification failed for {self.index_dir}: {e}")
            paper_type = "literature_review"
            return "data_analysis"  
        return paper_type
    
    def _extract_data_sources(self, paper_type: str, index_path: str, chunks_path: str, 
                            references, pdf_path, metadata):
        """Extract data sources using the simplified search strategy."""
        
        if paper_type == "literature_review":
            # Special handling for literature reviews
            table_image_results = self._extract_literature_review_sources(pdf_path, references)

            # query paper to see if there is a link to appendix or suppl data (with info on papers used for review)
            # appendix_info = self._query_appendix_review(pdf_path)

            # return extraction_result, __None # RETURN LATER TO ALLOW FOR SEARCHING SUPPL DATA

        # Generate queries for the paper type
        queries = QueryGenerator.generate_queries(paper_type, metadata, self.hyde_generate)
        
        # Perform multi-strategy search
        all_results = []

        # Semantic search
        if index_path:
            semantic_results = self.chunk_searcher.search_semantic(queries, index_path, chunks_path)
            all_results.extend(semantic_results)
        
        # Keyword search
        keyword_results = self.chunk_searcher.search_keyword(chunks_path)
        all_results.extend(keyword_results)
        
        # Context extraction
        context_results = self.chunk_searcher.extract_context_snippets(chunks_path)
        all_results.extend(context_results)
        
        # Deduplicate and rank results
        final_results = ResultRanker.deduplicate_and_rank(all_results, self.config.search.top_k_final)

        # Convert to format expected by extractor
        relevant_chunks = []
        for result in final_results:
            chunk = {
                "id": result.id,
                "text": result.text,
                "section_type": result.section_type,
                "title": result.title,
                "similarity_score": result.similarity_score,
                "rank_score": result.rank_score,
                "artifacts": result.artifacts
            }
            relevant_chunks.append(chunk)
        
        # Extract using LLM
        extraction_query = self._create_extraction_query(paper_type, metadata)
        
        
        try:
            extraction_result, confidence_scores = self.extractor.extract_data_sources(
                relevant_chunks=relevant_chunks,
                references=references,
                paper_type=paper_type,
                metadata=metadata,
                query=extraction_query
            )
            if paper_type == "literature_review":
                extraction_result.add_matched_dois(table_image_results.get("matched_dois", []))
            return extraction_result, confidence_scores
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {"data_sources": [], "data_sources_description": "Extraction failed"}, None
    
    def _create_extraction_query(self, paper_type: str, metadata) -> str:
        """Create extraction query based on paper type."""
        base_query = (
            "Extract all explicit data availability statements, URLs, DOIs, accession numbers, "
            "and repository information. Classify availability as: 'open', 'restricted', "
            "'upon_request', or 'not_specified'."
        )
        
        if hasattr(metadata, 'title') and metadata.title:
            base_query += f" Paper title: '{metadata.title}'"
        
        return base_query
    

    # TODO !!! 
    def _extract_literature_review_sources(self, pdf_path: str, references):
        """Special extraction for literature reviews using table/figure extraction."""
        # try:

        from .figure_table_extractor import TableExtractor 
        base_dir = Path(pdf_path).parent
        fte = TableExtractor(pdf_path=pdf_path, output_root=base_dir)
        table_image_results = fte.run_with_matcher(references)
        return table_image_results
            # extraction_result = {
            #     "data_sources": table_image_results.get("text_tables", []),
            #     "images": table_image_results.get("images", []),
            #     "image_candidates": table_image_results.get("image_table_candidates", []),
            #     "data_sources_description": "---",
            # }
            # return extraction_result, None # THIS 
     
        # except Exception as e:
        #     logger.warning(f"Literature review extraction failed: {e}")
        #     return {"data_sources": [], "data_sources_description": "Extraction failed"}, None
        

    def _create_result_dict(self, paper_id: str, pdf_path: str, metadata, sections, references, analysis_type: str, extraction_result, confidence_scores) -> Dict[str, Any]:
        references_list = []
        if extraction_result and hasattr(extraction_result, "references"):
            for r in extraction_result.references:
                if hasattr(r, 'model_dump'):
                    references_list.append(r.model_dump())
                elif isinstance(r, dict):
                    references_list.append(r)
        
        # A safer way to handle 'data_sources'
        data_sources_list = []
        if extraction_result and hasattr(extraction_result, "data_sources"):
            for ds in extraction_result.data_sources:
                if hasattr(ds, 'model_dump'):
                    data_sources_list.append(ds.model_dump())
                elif isinstance(ds, dict):
                    data_sources_list.append(ds)

        return {
            "paper_id": paper_id,
            "paper_title": getattr(metadata, "title", None),
            "first_author": getattr(metadata, "first_author", None),
            "publication_year": getattr(metadata, "publication_year", None),
            "doi": getattr(metadata, "doi", None),
            "journal": getattr(metadata, "journal", None),
            "abstract": (getattr(metadata, "abstract", None) or ""),
            "keywords": json.dumps(getattr(metadata, "keywords", []) or []),
            "analysis_type": analysis_type,
            "num_sections": len(sections),
            "num_references": len(references),
            "data_source_references": sum(1 for r in references if getattr(r, "is_data_source", False)),
            "references": json.dumps(references_list),
            "data_sources_description": getattr(extraction_result, "data_sources_description", "---") if extraction_result else "---",
            "data_sources": json.dumps(data_sources_list),
            "pdf_path": pdf_path,
            "confidence_scores": json.dumps(confidence_scores) if confidence_scores else None,
        }    

            
    def _create_result_markdown(
        self,
        paper_id: str,
        pdf_path: str,
        metadata,
        sections,
        references,
        analysis_type: str,
        extraction_result,
        confidence_scores
    ) -> str:
        """Create a markdown summary of the extraction result."""
        md = []
        md.append(f"# Paper ID: {paper_id}\n")
        md.append(f"**Title:** {getattr(metadata, 'title', 'N/A')}\n")
        md.append(f"**First Author:** {getattr(metadata, 'first_author', 'N/A')}\n")
        md.append(f"**Publication Year:** {getattr(metadata, 'publication_year', 'N/A')}\n")
        md.append(f"**DOI:** {getattr(metadata, 'doi', 'N/A')}\n")
        md.append(f"**Journal:** {getattr(metadata, 'journal', 'N/A')}\n")
        md.append(f"**Analysis Type:** {analysis_type}\n")
        md.append(f"**PDF Path:** {pdf_path}\n")
        md.append("\n---\n")
        
        md.append("## Abstract\n")
        abstract = getattr(metadata, "abstract", "N/A") or "N/A"
        md.append(f"{abstract}\n")
        
        md.append("## Keywords\n")
        keywords = getattr(metadata, "keywords", []) or []
        if keywords:
            md.append(", ".join(keywords) + "\n")
        else:
            md.append("N/A\n")
        
        # md.append("## Sections\n")
        # for section in sections:
        #     if isinstance(section, dict):
        #         title = section.get("title", "Untitled")
        #         sec_type = section.get("section_type", "other")
        #         content = section.get("content", "")
        #     else:  # e.g. StructuredSection object
        #         title = getattr(section, "title", "Untitled")
        #         sec_type = getattr(section, "section_type", "other")
        #         content = getattr(section, "content", "")
        #     content_preview = content.replace("\n", " ")
                            
        #     md.append(f"- **{title}** ({sec_type}): {content_preview}\n")
        
        # md.append("\n---\n")
        
        # md.append("## References\n")
        # for ref in references:
        #     if isinstance(ref, dict):
        #         ref_str = ref.get("raw_reference", "")
        #         is_data_source = ref.get("is_data_source", False)
        #     else:
        #         ref_str = getattr(ref, "raw_reference", "")
        #         is_data_source = getattr(ref, "is_data_source", False)
        #     md.append(f"- {'[DATA SOURCE] ' if is_data_source else ''}{ref_str}\n")
        
        # md.append("\n---\n")
        
        md.append("## Extracted Data Sources\n")
        if extraction_result and hasattr(extraction_result, "data_sources"):
            for ds in extraction_result.data_sources:
                if isinstance(ds, dict):
                    source_name = ds.get("source_name", "N/A")
                    url = ds.get("url", "N/A")
                    explanation = ds.get("explanation", "N/A")
                    section_found = ds.get("section_found", "N/A")
                else: # e.g., DataSource object
                    source_name = getattr(ds, "source_name", "N/A")
                    url = getattr(ds, "url", "N/A")
                    explanation = getattr(ds, "explanation", "N/A")
                    section_found = getattr(ds, "section_found", "N/A")
                md.append(f"- **Source Name:** {source_name}\n")
                md.append(f"  - **URL:** {url}\n")
                md.append(f"  - **Explanation:** {explanation}\n")
                md.append(f"  - **Section Found:** {section_found}\n\n")
        else:
            md.append("No data sources extracted.\n")   
        
        md.append("\n---\n")
        md.append("## Data Sources Description\n")
        description = getattr(extraction_result, "data_sources_description", "N/A") if extraction_result else "N/A"
        md.append(f"{description}\n")   
        
        if confidence_scores:
            md.append("\n---\n")
            md.append("## Confidence Scores\n")
            for key, score in confidence_scores.items():
                md.append(f"- **{key}:** {score}\n\n")


        if extraction_result and hasattr(extraction_result, "matched_dois"):
            md.append("\n---\n")
            md.append("## DOIs\n")
            for doi in extraction_result.matched_dois:
                md.append(f"- https://doi.org/{doi}\n")

        return "\n".join(md)


    def process_directory(self, input_dir: str, strategy_name: str, max_files: Optional[int] = None) -> None:
        """Process all PDFs in a directory."""
        pdfs = list(Path(input_dir).rglob("*.pdf"))
        if max_files:
            # pdfs = [p for p in pdfs if GT_PAPER_TYPES.get(p.stem, "") == SELECTED_TYPE]
            # print([p.stem for p in pdfs])
            pdfs = pdfs[:max_files]
        logger.info(f"Processing {len(pdfs)} PDFs with {self.config.max_workers} workers")

        md_final_filename = self.output_dir / "final_extraction.md"
        md_final_content = ""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.process_pdf, str(pdf), strategy_name=strategy_name): pdf for pdf in pdfs}
            
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    result_csv, result_markdown = future.result()
                    self.results.append(result_csv)
                    
                    # Save incremental results - csv
                    df = pd.DataFrame([result_csv])
                    df.to_csv(self.results_file, mode="a", header=not self.results_file.exists(), index=False)

                    # # Save incremental results - markdown
                    # md_file = self.output_dir / f"{pdf_path.stem}_extraction.md"
                    # with open(md_file, "a", encoding="utf-8") as f:
                    #     f.write(result_markdown)

                    md_final_content += result_markdown

                    logger.info(f"✓ {pdf_path.name} completed successfully")
                    
                except Exception as e:
                    logger.error(f"✗ {pdf_path.name} failed: {e}")
        
        # Save final results
        if self.results:
            # writing csv output
            final_file = self.output_dir / "_final_results.csv"
            pd.DataFrame(self.results).to_csv(final_file, index=False)
            # writing markdown output
            md_final_filename = self.output_dir / "final_extraction.md"
            with open(md_final_filename, "w", encoding="utf-8") as f:
                f.write(md_final_content)

            logger.info(f"Saved final results to {final_file}, {md_final_filename}")
            self._print_summary()
    
    def _print_summary(self) -> None:
        """Print processing summary."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        logger.info("=== Processing Summary ===")
        logger.info(f"Total papers processed: {len(df)}")
        
        if "analysis_type" in df.columns:
            logger.info("Analysis type distribution:")
            for analysis_type, count in df["analysis_type"].value_counts().items():
                logger.info(f"  {analysis_type}: {count}")















