import sys
import types
import unittest
from unittest.mock import patch, MagicMock
import tempfile

# # --- Ensure a FAISS stub exists BEFORE importing your code under test ---
# if "faiss" not in sys.modules:
#     sys.modules["faiss"] = types.SimpleNamespace(
#         # include whatever your code might touch at import time
#         Index=type("Index", (), {}),
#         IndexFlatL2=type("IndexFlatL2", (), {}),
#     )

from episcope.storage.in_memory_academic_db import InMemoryAcademicDB
from episcope.core.blueprints.data_blueprints import PaperMetadata, StructuredSection
from episcope.pipelines.precision_miner_pipeline import PipelineProcessor
# from episcope.config import Config, SearchConfig


class TestPipelineProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.db = InMemoryAcademicDB()
        self.output_dir = tempfile.TemporaryDirectory()
        self.index_dir = tempfile.TemporaryDirectory()

        # self.config = Config(
        #     embedding_model="all-MiniLM-L6-v2",
        #     classifier_model="tinyllama:1.1b",
        #     llm_model="tinyllama:1.1b",
        #     hyde_model="tinyllama:1.1b",
        #     use_hyde=False,
        #     max_workers=1,
        #     search=SearchConfig(top_k_final=10),
        # )

    def tearDown(self) -> None:
        self.output_dir.cleanup()
        self.index_dir.cleanup()

    @patch("episcope.pipelines.precision_miner_pipeline.EmbeddingIndexer")
    @patch("episcope.pipelines.precision_miner_pipeline.PaperClassifier")
    @patch("episcope.pipelines.precision_miner_pipeline.RAGQuerier")
    @patch("episcope.pipelines.precision_miner_pipeline.LLMExtractor")
    @patch("episcope.retrieve.embeddings.SimplifiedEmbedder")
    def test_run_pipeline(self, MockSimplifiedEmbedder, MockLLMExtractor, MockRAGQuerier, MockPaperClassifier, MockEmbeddingIndexer):
        # Setup mocks
        mock_embedder = MockSimplifiedEmbedder.return_value
        mock_indexer = MockEmbeddingIndexer.return_value
        mock_classifier = MockPaperClassifier.return_value
        mock_querier = MockRAGQuerier.return_value
        mock_extractor = MockLLMExtractor.return_value

        mock_indexer.create_index.return_value = "dummy_index_path"
        
        class MockClassificationResult:
            class MockPaperType:
                value = "data_analysis"
            paper_type = MockPaperType()
        
        mock_classifier.get_relevant_chunks.return_value = {}
        mock_classifier.classify_based_on_relevant_chunks.return_value = MockClassificationResult()
        
        mock_extractor.generate.return_value = (MagicMock(), {})

        # Populate DB
        paper_id = "test_paper"
        strategy_name = "test_strategy"
        pdf_path = "/fake/path/test_paper.pdf"
        
        metadata = PaperMetadata(title="Test Paper", file_path=pdf_path)
        sections = [StructuredSection(title="Intro", content="This is the introduction.")]
        
        self.db.insert(paper_id, "metadata", strategy_name, metadata.to_dict())
        self.db.insert(paper_id, "sections", strategy_name, [s.to_dict() for s in sections])
        self.db.insert(paper_id, "references", strategy_name, [])

        # Initialize and run processor
        processor = PipelineProcessor(db=self.db, output_dir=self.output_dir.name, index_dir=self.index_dir.name, config=self.config)
        processor.run(strategy_name=strategy_name)

        # Assertions
        self.assertEqual(len(processor.results), 1)
        result_dict = processor.results[0]
        self.assertEqual(result_dict["paper_id"], paper_id)
        self.assertEqual(result_dict["paper_title"], "Test Paper")
        self.assertEqual(result_dict["analysis_type"], "data_analysis")
        
        mock_indexer.create_index.assert_called_once()
        mock_classifier.classify_based_on_relevant_chunks.assert_called_once()
        mock_extractor.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
