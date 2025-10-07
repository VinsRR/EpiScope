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


class TestSimplifiedEmbedder(unittest.TestCase):
    """Tests for the simplified embedding wrapper."""

    def test_embed_text_uses_underlying_embedder(self) -> None:
        # Create dummy embedding model
        class DummyHF:
            def __init__(self, *args, **kwargs):
                pass
            def _get_query_embedding(self, text: str):
                # Return a constant vector whose length depends on the input
                return [len(text)]
            def get_text_embedding_batch(self, texts):
                return [[len(t)] for t in texts]

        class DummyConfig:
            hidden_size = 1

        with patch("episcope.retrieve.embeddings.HuggingFaceEmbedding", return_value=DummyHF()) as mock_hf, \
             patch("episcope.retrieve.embeddings.AutoConfig.from_pretrained", return_value=DummyConfig()):
            embedder = SimplifiedEmbedder(embed_model="dummy-model", batch_size=2)
            # Single text
            vec = embedder.embed_text("hello")
            self.assertEqual(vec, [5])
            # Batch texts
            vecs = embedder.embed_texts(["hi", "world"])
            self.assertEqual(vecs, [[2], [5]])
            # Embedding dimension should reflect DummyConfig.hidden_size
            self.assertEqual(embedder.dim, 1)


if __name__ == "__main__":
    unittest.main()