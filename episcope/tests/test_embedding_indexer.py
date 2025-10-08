"""
Tests for the EmbeddingIndexer class.

These tests verify that the `create_index` method correctly generates
and saves a FAISS index and a structured chunks file.
"""
import unittest
import os
import json
from unittest import mock

from .configs import test_model_hf_embedding as embed_model

# Stub external dependencies before imports
import sys
import types
sys.modules.setdefault(
    "faiss",
    types.SimpleNamespace(
        read_index=lambda *args, **kwargs: None,
        write_index=lambda *args, **kwargs: None,
        IndexFlatIP=lambda *args, **kwargs: types.SimpleNamespace(add=lambda x: None),
        normalize_L2=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "episcope.retrieve.embeddings",
    types.SimpleNamespace(
        SimplifiedEmbedder=lambda *args, **kwargs: types.SimpleNamespace(
            embed_text=lambda t: [len(t)],
            embed_texts=lambda t_list: [[len(t)] for t in t_list]
        )
    ),
)

from episcope.parse.indexing.embeddings import EmbeddingIndexer
from episcope.parse.blueprints.data_blueprints import StructuredSection, PaperMetadata

class TestEmbeddingIndexer(unittest.TestCase):
    """Test cases for EmbeddingIndexer file creation."""

    def setUp(self) -> None:
        self.indexer = EmbeddingIndexer(model_name=embed_model, min_chunk_size=10)
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self) -> None:
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @mock.patch("faiss.write_index")
    @mock.patch("os.path.exists", side_effect=lambda p: p.endswith(".faiss") or p.endswith(".json")) # Mock existence of files.
    def test_create_index_files(self, mock_exists, mock_write_index) -> None:
        """Ensure create_index creates both index and chunks files."""
        sections = [
            StructuredSection(title="Intro", content="This is the introduction with more than five words.", section_type="introduction"),
            StructuredSection(title="Methods", content="This is the methods section, also with more than five words.", section_type="methods")
        ]
        metadata = PaperMetadata(title="Test Paper", abstract="This is the abstract.")
        paper_id = "test_paper_123"

        index_path = self.indexer.create_index(sections, metadata, self.test_dir, paper_id)

        # despite the mock, ensure the paths are as expected
        expected_index_path = os.path.join(self.test_dir, f"{paper_id}_structured_index.faiss")
        self.assertEqual(index_path, expected_index_path)
        self.assertTrue(mock_exists(index_path))
        #
        chunks_path = os.path.join(self.test_dir, f"{paper_id}_structured_chunks.json")
        self.assertTrue(mock_exists(chunks_path))

        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)
        
        self.assertEqual(len(chunks_data), 3) # Abstract + 2 sections
        self.assertEqual(chunks_data[0]['section_title'], "Abstract")
        self.assertEqual(chunks_data[1]['text'], "This is the introduction with more than five words.")
        
        mock_write_index.assert_called_once_with(mock.ANY, expected_index_path)
                                                 
if __name__ == "__main__":
    unittest.main()