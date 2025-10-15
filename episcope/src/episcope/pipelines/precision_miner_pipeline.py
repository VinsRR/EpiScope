import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from episcope.db.academic_db import AcademicDB
from episcope.pipelines.base import BatchRAGPipeline
from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.rag.generation.llm_extractor import LLMExtractor
from episcope.rag.indexing.indexer import Indexer
from episcope.rag.postprocessing.artifact_extractor import ArtifactExtractor
from episcope.rag.postprocessing.context_extractor import ContextualSnippetRetriever
from episcope.rag.postprocessing.reranker import ResultRanker
from episcope.rag.retrieval.keyword import KeywordRetriever
from episcope.rag.retrieval.semantic import SemanticRetriever
from episcope.vectordb.file import FileDB
from episcope.utils.data_blueprints import PaperMetadata, StructuredSection
from episcope.parse_configs import PipelineConfig
from episcope.utils.processing_utils import QueryGenerator, TextProcessor

logger = logging.getLogger(__name__)

class PipelineProcessor(BatchRAGPipeline):
    """Main pipeline processor with a modular retrieval and post-processing architecture."""

    def __init__(self, db: AcademicDB, output_dir: str = "output", index_dir: str = "indices",
                 config: Optional[PipelineConfig] = None):
        self.db = db
        self.output_dir = Path(output_dir)
        self.index_dir = Path(index_dir)
        self.config = config or PipelineConfig()

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.index_dir.mkdir(exist_ok=True, parents=True)

        self._init_components()

        self.results = []
        self.results_file = self.output_dir / "_extraction_results.csv"

    def _init_components(self):
        """Initialize all pipeline components."""
        from episcope.pipelines.classification_pipeline import PaperClassifier

        embedder = SimplifiedEmbedder(embed_model=self.config.embedding_model)
        self.vector_db = FileDB(index_dir=str(self.index_dir))
        self.indexer = Indexer(db=self.vector_db, embed_model=self.config.embedding_model)
        self.text_processor = TextProcessor()

        # Retrievers
        self.semantic_retriever = SemanticRetriever(self.vector_db)
        self.keyword_retriever = KeywordRetriever(self.vector_db)
        self.context_retriever = ContextualSnippetRetriever(self.vector_db, self.text_processor)

        # Post-processors
        self.artifact_extractor = ArtifactExtractor(self.text_processor)
        self.reranker = ResultRanker()

        # Other components
        self.paper_classifier = PaperClassifier(self.config.classifier_model, embedder=embedder, vectordb=self.vector_db)
        self.extractor = LLMExtractor(self.config.llm_model)

        self.hyde_generate = None
        if self.config.use_hyde:
            try:
                from episcope.src.episcope.utils.hyde import HYDE
                hyde = HYDE(self.config.hyde_model)
                self.hyde_generate = hyde.generate_multiple
            except ImportError:
                logger.debug("HyDE not available")

    def process_item(self, paper_id: str, strategy_name: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Process a single paper."""
        logger.info(f"Processing paper: {paper_id}")
        
        paper_data = self.db.get_paper(paper_id, strategy_name)
        if not paper_data:
            logger.warning(f"Could not retrieve data for paper {paper_id}")
            return None

        metadata = PaperMetadata(**paper_data.get("metadata", {}))
        sections = [StructuredSection(**s) for s in paper_data.get("sections", [])]
        references = paper_data.get("references", [])
        pdf_path = paper_data.get("pdf_path")

        self.indexer.index_paper(sections=sections, metadata=metadata, paper_id=paper_id)
        paper_type = self._classify_paper_type(metadata, paper_id)
        
        extraction_result, confidence = self._extract_data_sources(
            paper_type, paper_id, references, pdf_path, metadata
        )

        result_csv = self._format_result_csv(paper_id, metadata, extraction_result, confidence)
        result_md = self._format_result_markdown(paper_id, metadata, extraction_result)
        
        paper_output_dir = self.output_dir / paper_id
        paper_output_dir.mkdir(exist_ok=True)
        with open(paper_output_dir / "extraction.json", "w") as f:
            json.dump(extraction_result, f, indent=2, default=str)
        with open(paper_output_dir / "result.md", "w") as f:
            f.write(result_md)

        return result_csv, result_md

    def _classify_paper_type(self, metadata: PaperMetadata, paper_id: str) -> str:
        """Classify the paper type."""
        try:
            relevant_chunks = self.paper_classifier.get_relevant_chunks(metadata, paper_id)
            classification_result = self.paper_classifier.classify_based_on_relevant_chunks(relevant_chunks, metadata)
            return classification_result.paper_type.value
        except Exception as e:
            logger.warning(f"Paper classification failed for {paper_id}: {e}")
            return "data_analysis"

    def _extract_data_sources(self, paper_type: str, paper_id: str, references: List[Any], pdf_path: Optional[str], metadata: PaperMetadata):
        """Extract data sources using the modular retrieval and post-processing pipeline."""
        table_image_results = {}
        if paper_type == "literature_review" and pdf_path:
            table_image_results = self._extract_literature_review_sources(pdf_path, references)

        queries = QueryGenerator.generate_queries(paper_type, metadata, self.hyde_generate)
        
        # --- Retrieval Stage ---
        all_results = []
        for query, _, _ in queries:
            semantic_results = self.semantic_retriever.retrieve(query, paper_id=paper_id, top_k=self.config.search.top_k_semantic)
            all_results.extend(semantic_results)

        keyword_query = "data availability available repository accession deposited supplementary github zenodo"
        keyword_results = self.keyword_retriever.retrieve(keyword_query, paper_id=paper_id, top_k=self.config.search.top_k_keyword)
        all_results.extend(keyword_results)
        
        context_results = self.context_retriever.retrieve("", paper_id=paper_id, top_k=100) # top_k is high to get all snippets
        all_results.extend(context_results)
        
        # --- Post-processing Stage ---
        processed_results = self.artifact_extractor.process(all_results)
        final_results = self.reranker.rerank(processed_results, self.config.search.top_k_final)

        relevant_chunks = [result.__dict__ for result in final_results]
        
        extraction_query = self._create_extraction_query(paper_type, metadata)
        
        try:
            extraction_result, confidence_scores = self.extractor.generate(
                relevant_chunks=relevant_chunks,
                references=references,
                paper_type=paper_type,
                metadata=metadata,
                query=extraction_query
            )
            if paper_type == "literature_review" and "add_matched_dois" in dir(extraction_result):
                extraction_result.add_matched_dois(table_image_results.get("matched_dois", []))
            return extraction_result, confidence_scores
            
        except Exception as e:
            logger.error(f"LLM extraction failed for {paper_id}: {e}")
            return {"data_sources": [], "data_sources_description": "Extraction failed"}, None

    def _create_extraction_query(self, paper_type: str, metadata: PaperMetadata) -> str:
        """Create extraction query based on paper type."""
        base_query = (
            "Extract all explicit data availability statements, URLs, DOIs, accession numbers, "
            "and repository information. Classify availability as: 'open', 'restricted', "
            "'upon_request', or 'not_specified'."
        )
        if metadata.title:
            base_query += f" Paper title: '{metadata.title}'"
        return base_query

    def _extract_literature_review_sources(self, pdf_path: str, references: List[Any]):
        """Special extraction for literature reviews using table/figure extraction."""
        from episcope.pipelines.figure_table_extractor import TableExtractor
        base_dir = Path(pdf_path).parent
        fte = TableExtractor(pdf_path=pdf_path, output_root=base_dir)
        return fte.run_with_matcher(references)

    def run(self, items: List[Any]) -> List[Any]:
        """Run the pipeline on a batch of items."""
        paper_ids = items
        strategy_name = "default"

        logger.info(f"Processing {len(paper_ids)} papers with {self.config.max_workers} workers for strategy '{strategy_name}'")

        md_final_filename = self.output_dir / f"{strategy_name}_extraction.md"
        md_final_content = ""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_paper_id = {executor.submit(self.process_item, paper_id, strategy_name): paper_id for paper_id in paper_ids}
            for future in as_completed(future_to_paper_id):
                paper_id = future_to_paper_id[future]
                try:
                    result = future.result()
                    if result:
                        result_csv, result_markdown = result
                        self.results.append(result_csv)
                        df = pd.DataFrame([result_csv])
                        df.to_csv(self.results_file, mode="a", header=not self.results_file.exists(), index=False)
                        md_final_content += result_markdown
                        logger.info(f"✓ {paper_id} completed successfully")
                except Exception as exc:
                    logger.error(f"✗ {paper_id} generated an exception: {exc}")

        if self.results:
            final_file = self.output_dir / f"_{strategy_name}_final_results.csv"
            pd.DataFrame(self.results).to_csv(final_file, index=False)
            with open(md_final_filename, "w", encoding="utf-8") as f:
                f.write(md_final_content)
            logger.info(f"Saved final results to {final_file} and {md_final_filename}")
        
        return self.results

    def _format_result_csv(self, paper_id, metadata, extraction_result, confidence):
        return {
            "paper_id": paper_id,
            "title": metadata.title,
            "description": extraction_result.get("data_sources_description", ""),
            "data_sources": json.dumps(extraction_result.get("data_sources", [])),
            "confidence": json.dumps(confidence)
        }

    def _format_result_markdown(self, paper_id, metadata, extraction_result):
        md = f"## Paper: {paper_id} - {metadata.title}\n\n"
        md += f"**Description:** {extraction_result.get('data_sources_description', 'N/A')}\n\n"
        md += "**Data Sources:**\n"
        sources = extraction_result.get("data_sources", [])
        if sources:
            for source in sources:
                md += f"- **{source.get('source_name', 'N/A')}**: {source.get('explanation', '')} ({source.get('url', 'No URL')})\n"
        else:
            md += "- None found\n"
        md += "\n---\n"
        return md
