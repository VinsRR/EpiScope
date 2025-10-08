"""
Unit tests for paper classifier utilities.

These tests exercise internal helper methods of the
``PaperClassifier`` class to ensure they behave deterministically.
Heavy dependencies such as FAISS, Ollama and SentenceTransformer are
not invoked here; instead we focus on pure Python logic like
deduplication and ranking of chunks.
"""

# The "dummy" embedding model could be used by using unittest.mock to stub
# out the SentenceTransformer class: https://docs.python.org/3/library/unittest.mock.html

import unittest

# Stub external modules that are unavailable in the test environment.
import sys
import types
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ),
)
sys.modules.setdefault(
    "faiss",
    types.SimpleNamespace(
        read_index=lambda *args, **kwargs: None,
        write_index=lambda *args, **kwargs: None,
        IndexFlatIP=lambda *args, **kwargs: None,
        normalize_L2=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "ollama",
    types.SimpleNamespace(
        Client=lambda *args, **kwargs: types.SimpleNamespace(chat=lambda *a, **kw: None),
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

from episcope.parse.extraction.paper_classifier import PaperClassifier
from episcope.retrieve.embeddings import SimplifiedEmbedder
from .configs import test_model_hf_embedding, test_model_ollama

model_name_emb = test_model_hf_embedding
model_name = test_model_ollama
class TestPaperClassifier(unittest.TestCase):
    """Tests for helper functions in PaperClassifier."""

    def test_deduplicate_and_rank_chunks(self) -> None:
        """Ensure deduplication keeps the highest score and sorts descending."""
        mock_embedder = SimplifiedEmbedder(embed_model=model_name_emb)
        classifier = PaperClassifier(
            model_name=model_name,
            embedder=mock_embedder
        )
        aggregated = {
            "literature_review": [
                ("duplicate text", 0.3),
                ("duplicate text", 0.2),
                ("unique text", 0.8),
                ("another", 0.1),
            ],
            "data_analysis": []
        }
        top = classifier._deduplicate_and_rank_chunks(aggregated, top_k=2)
        # For literature_review, expect two entries: unique text (0.8) and duplicate text (0.3)
        self.assertEqual(len(top["literature_review"]), 2)
        texts = [t for t, _ in top["literature_review"]]
        self.assertIn("unique text", texts)
        self.assertIn("duplicate text", texts)
        # Scores should be in descending order
        scores = [s for _, s in top["literature_review"]]
        self.assertGreaterEqual(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()