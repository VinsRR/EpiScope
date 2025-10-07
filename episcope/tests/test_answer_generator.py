"""
Tests for the high‑level answer generation pipeline.

These tests patch the RAG factory and provider factory to avoid
connecting to external services.  They validate that the
AnswerGenerator combines retrieved contexts into a provenance object
with the correct answer and evidences.
"""
import unittest
from unittest.mock import patch

import sys
import types

# Attempt to import AnswerGenerator; if dependencies are missing, skip the test.
try:
    from episcope.generate.answer import AnswerGenerator  # type: ignore
    _IMPORT_ERROR = False
except Exception:
    # Do not raise here; record that import failed so we can skip tests.
    AnswerGenerator = None  # type: ignore
    _IMPORT_ERROR = True


@unittest.skipIf(_IMPORT_ERROR or AnswerGenerator is None, "AnswerGenerator dependencies unavailable; skipping")
class TestAnswerGenerator(unittest.TestCase):
    """Tests for AnswerGenerator.answer_question."""

    def test_answer_question_returns_provenance(self) -> None:
        # Dummy retrieval contexts
        dummy_contexts = [
            {"content": "Snippet one.", "id": "paper1", "section_type": "Methods"},
            {"content": "Snippet two.", "id": "paper2", "section_type": "Results"},
        ]
        # Dummy rag object with retrieve method
        class DummyRAG:
            system_prompt = "System prompt"
            user_prompt = "Answer the question: {question}\n\n{snippet_list}"
            def retrieve(self, query: str, top_k: int = 5):
                return dummy_contexts

        # Dummy provider returning a fixed answer
        class DummyProvider:
            def chat(self, messages):
                return "The answer to your question."

        with patch("episcope.generate.answer.RAGFactory.get", return_value=DummyRAG()), \
             patch("episcope.generate.answer.ProviderFactory.create", return_value=DummyProvider()):
            ag = AnswerGenerator(rag_method="text")
            provenance = ag.answer_question("What is the question?")
            # The answer should match the dummy provider response
            self.assertEqual(provenance.answer, "The answer to your question.")
            # There should be one evidence per context
            self.assertEqual(len(provenance.evidences), 2)
            # Evidence fields should be populated
            self.assertEqual(provenance.evidences[0].paper_id, "paper1")
            self.assertEqual(provenance.evidences[1].paper_id, "paper2")


if __name__ == "__main__":
    unittest.main()