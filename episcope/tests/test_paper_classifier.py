"""
Unit tests for paper classifier utilities.

These tests exercise internal helper methods of the
``PaperClassifier`` class to ensure they behave deterministically.
"""
import unittest
from unittest.mock import MagicMock

from episcope.workflows.classification_workflow import PaperClassifier
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever


class MockRetriever(AbstractRetriever):
    def retrieve(self, query: str, **kwargs):
        pass

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        pass

class MockGenerator(Generator):
    def generate(self, **kwargs):
        pass

class TestPaperClassifier(unittest.TestCase):
    """Tests for helper functions in PaperClassifier."""

    def test_deduplicate_and_rank_chunks(self) -> None:
        """Ensure deduplication keeps the highest score and sorts descending."""
        mock_retriever = MockRetriever()
        mock_generator = MockGenerator()
        classifier = PaperClassifier(
            retriever=mock_retriever,
            generator=mock_generator
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
