import unittest
import os
from unittest.mock import patch, MagicMock
import tempfile

from episcope.rag.indexing.specialized_faiss_indexer import EmbeddingIndexer
from episcope.utils.data_blueprints import StructuredSection, PaperMetadata

class TestEmbeddingIndexer(unittest.TestCase):
    @patch("episcope.rag.indexing.specialized_faiss_indexer.SimplifiedEmbedder")
    @patch("episcope.rag.indexing.specialized_faiss_indexer.faiss")
    def test_create_index_files(self, mock_faiss, mock_embedder) -> None:
        """Ensure create_index creates both index and chunks files."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Configure mocks
            mock_embedder.return_value.embed_texts.return_value = [[0.1, 0.2]]
            
            indexer = EmbeddingIndexer(model_name="dummy-model")

            sections = [
                StructuredSection(title="Intro", content="This is the introduction with more than five words.", section_type="introduction"),
                StructuredSection(title="Methods", content="This is the methods section, also with more than five words.", section_type="methods")
            ]
            metadata = PaperMetadata(title="Test Paper", abstract="This is the abstract.")
            paper_id = "test_paper_123"
        
            index_path = indexer.create_index(sections, metadata, test_dir, paper_id)
        
            # Assertions
            expected_index_path = os.path.join(test_dir, f"{paper_id}_structured_index.faiss")
            self.assertEqual(index_path, expected_index_path)
            
            # Verify that the chunks file was created
            chunks_path = os.path.join(test_dir, f"{paper_id}_structured_chunks.json")
            self.assertTrue(os.path.exists(chunks_path))
            
            # Verify faiss.write_index was called
            mock_faiss.write_index.assert_called_once()

if __name__ == "__main__":
    unittest.main()
