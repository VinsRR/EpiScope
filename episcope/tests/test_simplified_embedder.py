"""
Tests for the SimplifiedEmbedder class.

These tests ensure that the embedder correctly delegates to the
underlying HuggingFace embedding implementation.  We patch the
HuggingFaceEmbedding and AutoConfig classes so that no external
models are downloaded or initialised during the test run.
"""
import unittest
from unittest.mock import patch

import types
import sys
sys.modules.setdefault(
    "qdrant_client",
    types.ModuleType("qdrant_client"),
)
sys.modules.setdefault(
    "qdrant_client.http",
    types.ModuleType("qdrant_client.http"),
)
sys.modules.setdefault(
    "qdrant_client.http.models",
    types.ModuleType("qdrant_client.http.models"),
)
sys.modules.setdefault(
    "llama_index",
    types.ModuleType("llama_index"),
)
sys.modules.setdefault(
    "llama_index.embeddings",
    types.ModuleType("llama_index.embeddings"),
)
sys.modules.setdefault(
    "llama_index.embeddings.huggingface",
    types.ModuleType("llama_index.embeddings.huggingface"),
)

from episcope.retrieve.embeddings import SimplifiedEmbedder


class DummyHF:
    def __init__(self, *args, **kwargs):
        pass
    def _get_query_embedding(self, text: str):
        # Return a constant vector whose length depends on the input
        return [len(text)]
    def get_text_embedding_batch(self, texts):
        return [[len(t)] for t in texts]

class TestSimplifiedEmbedder(unittest.TestCase):
    def test_embed_text_uses_underlying_embedder(self) -> None:
        class DummyConfig:
            hidden_size = 1
    
        with patch("episcope.retrieve.embeddings.HuggingFaceEmbedding", return_value=DummyHF()) as mock_hf, \
             patch("episcope.retrieve.embeddings.AutoConfig.from_pretrained", return_value=DummyConfig()):
            
            embedder = SimplifiedEmbedder(embed_model="dummy-model")
            embedding = embedder.embed_text("hello")
            self.assertEqual(embedding, [5])
            
            embeddings = embedder.embed_texts(["hello", "world"])
            self.assertEqual(embeddings, [[5], [5]])


if __name__ == "__main__":
    unittest.main()