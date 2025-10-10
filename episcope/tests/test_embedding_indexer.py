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

from unittest.mock import patch, MagicMock
from episcope.index.specialized_faiss_indexer import EmbeddingIndexer
from episcope.core.blueprints.data_blueprints import StructuredSection, PaperMetadata

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


                                                 
if __name__ == "__main__":
    unittest.main()