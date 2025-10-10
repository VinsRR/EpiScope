import os
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from ..storage.academic_db import AcademicDB
from ..core.blueprints.data_blueprints import StructuredSection, Reference, PaperMetadata
from ..configs.parse_configs import PipelineConfig
from ..index.specialized_faiss_indexer import EmbeddingIndexer
from ..retrieve.chunk_searcher import ChunkSearcher
from ..retrieve.rerankers import ResultRanker
from ..utils.processing_utils import TextProcessor, QueryGenerator
from ..utils.result_utils import create_result_dict, create_result_markdown
from .base import BatchRAGPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class PipelineProcessor(BatchRAGPipeline):
    """Main pipeline processor with simplified architecture."""
    
    def __init__(self, db: AcademicDB, output_dir: str = "output", index_dir: str = "indices", 
                 config: Optional[PipelineConfig] = None):
        self.db = db
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
        from ..pipelines.classification_pipeline import PaperClassifier
        from ..retrieve.specialized_retriever import RAGQuerier
        from ..generate.llm_extractor import LLMExtractor
        from ..retrieve.embeddings import SimplifiedEmbedder

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
                from ..core.hyde import HYDE
                hyde = HYDE(self.config.hyde_model)
                self.hyde_generate = lambda prompt: list(hyde.generate(prompt, n=4))
            except Exception:
                logger.debug("HyDE not available")
    
    def process_item(self, paper_id: str, strategy_name: str = "default") -> Dict[str, Any]:
        """Process a single paper from the database."""
        logger.info(f"Processing {paper_id}")
        
        # Load extracted data from DB
        metadata, sections, references = self._load_extracted_data(paper_id, strategy_name)
        pdf_path = metadata.file_path
        if not pdf_path:
            raise ValueError(f"PDF path not found in metadata for paper {paper_id}")

        # Create/load index and chunks
        index_path, chunks_path = self._prepare_index_and_chunks(sections, metadata, paper_id, Path(pdf_path).parent, strategy_name)
        
        # Classify paper type
        paper_type = self._classify_paper_type(metadata, index_path, chunks_path)
        
        # Extract data sources
        extraction_result, confidence_scores = self._extract_data_sources(
            paper_type, index_path, chunks_path, references, pdf_path, metadata
        )
        
        # Create result dictionary
        result_dict = create_result_dict(
            paper_id, pdf_path, metadata, sections, references, 
            paper_type, extraction_result, confidence_scores
        )

        result_markdown = create_result_markdown(paper_id, pdf_path, metadata, sections, references,
            paper_type, extraction_result, confidence_scores)

        logger.info(f"Completed processing {paper_id}")
        return result_dict, result_markdown
    
    def _load_extracted_data(self, paper_id: str, strategy_name: str):
        """Load metadata, sections, and references from the database."""
        try:
            metadata = self.db.retrieve(paper_id, "metadata", strategy_name)
            sections = self.db.retrieve(paper_id, "sections", strategy_name)
            references = self.db.retrieve(paper_id, "references", strategy_name)

            if not metadata or not sections:
                raise FileNotFoundError(f"Data not found in DB for paper {paper_id} with strategy {strategy_name}")

            return metadata, sections, references or []
            
        except Exception as e:
            logger.error(f"Failed to load extracted data for {paper_id} from DB: {e}")
            raise
    
    def _prepare_index_and_chunks(self, sections, metadata, paper_id: str, base_dir: Path, strategy_name: str):
        """Prepare index and chunks files."""
        index_path = None
        if self.indexer:
            try:
                index_path = self.indexer.create_index(sections, metadata, str(self.index_dir), paper_id)
            except Exception as e:
                logger.warning(f"Index creation failed for {paper_id}: {e}")
        
        strategy_dir = self.index_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True, parents=True)
        chunks_path = strategy_dir / f"{paper_id}_structured_chunks.json"

        if not chunks_path.exists():
            chunks = []
            for i, section in enumerate(sections):
                if isinstance(section, StructuredSection):
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
                    assert False, f"Unexpected section type: {type(section)}"

                chunks.append(chunk)
                
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        return str(index_path) if index_path else "", str(chunks_path)
    
    def _classify_paper_type(self, metadata, index_path: str, chunks_path: str) -> str:
        """Classify the paper type."""
        try:
            relevant_chunks = self.paper_classifier.get_relevant_chunks(metadata=metadata, index_path=str(index_path), chunks_path=str(chunks_path))
            classification_result = self.paper_classifier.classify_based_on_relevant_chunks(relevant_chunks, metadata)
            paper_type = classification_result.paper_type.value
        except Exception as e:
            logger.warning(f"Paper classification failed for {self.index_dir}: {e}")
            paper_type = "data_analysis"  
        return paper_type
    
    def _extract_data_sources(self, paper_type: str, index_path: str, chunks_path: str, 
                            references, pdf_path, metadata):
        """Extract data sources using the simplified search strategy."""
        
        if paper_type == "literature_review":
            table_image_results = self._extract_literature_review_sources(pdf_path, references)

        queries = QueryGenerator.generate_queries(paper_type, metadata, self.hyde_generate)
        
        all_results = []

        if index_path:
            semantic_results = self.chunk_searcher.search_semantic(queries, index_path, chunks_path)
            all_results.extend(semantic_results)
        
        keyword_results = self.chunk_searcher.search_keyword(chunks_path)
        all_results.extend(keyword_results)
        
        context_results = self.chunk_searcher.extract_context_snippets(chunks_path)
        all_results.extend(context_results)
        
        final_results = ResultRanker().rerank(all_results, self.config.search.top_k_final)

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
        
        extraction_query = self._create_extraction_query(paper_type, metadata)
        
        try:
            extraction_result, confidence_scores = self.extractor.generate(
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
    
    def _extract_literature_review_sources(self, pdf_path: str, references):
        """Special extraction for literature reviews using table/figure extraction."""
        from .figure_table_extractor import TableExtractor 
        base_dir = Path(pdf_path).parent
        fte = TableExtractor(pdf_path=pdf_path, output_root=base_dir)
        table_image_results = fte.run_with_matcher(references)
        return table_image_results
        
    def run(self, strategy_name: str, paper_ids: Optional[List[str]] = None) -> None:
        """Process all papers for a given strategy."""
        if not paper_ids:
            paper_ids = self.db.list_papers(strategy_name)
        
        logger.info(f"Processing {len(paper_ids)} papers with {self.config.max_workers} workers for strategy '{strategy_name}'")

        md_final_filename = self.output_dir / f"{strategy_name}_extraction.md"
        md_final_content = ""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.process_item, paper_id, strategy_name=strategy_name): paper_id for paper_id in paper_ids}
            
            for future in as_completed(futures):
                paper_id = futures[future]
                try:
                    result_csv, result_markdown = future.result()
                    self.results.append(result_csv)
                    
                    df = pd.DataFrame([result_csv])
                    df.to_csv(self.results_file, mode="a", header=not self.results_file.exists(), index=False)

                    md_final_content += result_markdown

                    logger.info(f"✓ {paper_id} completed successfully")
                    
                except Exception as e:
                    logger.error(f"✗ {paper_id} failed: {e}")
        
        if self.results:
            final_file = self.output_dir / f"_{strategy_name}_final_results.csv"
            pd.DataFrame(self.results).to_csv(final_file, index=False)
            with open(md_final_filename, "w", encoding="utf-8") as f:
                f.write(md_final_content)

            logger.info(f"Saved final results to {final_file}, {md_final_filename}")
            self._print_summary()
    

