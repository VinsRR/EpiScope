"""
Tests for utility functions in ``episcope.retrieve.utils``.

These tests stub out external dependencies so that the module can be
imported without the ``qdrant_client`` or other third‑party packages.
The functions under test do not rely on those dependencies directly.
"""
import unittest
from unittest import mock
import sys
import types


class TestRetrieveUtils(unittest.TestCase):
    def setUp(self) -> None:
        # Prepare dummy modules for dependencies that may be imported by
        # ``episcope.rag.retrieval.utils``.  Only minimal attributes are
        # provided to satisfy import statements.
        self.dummy_modules = {
            "qdrant_client": types.SimpleNamespace(http=types.SimpleNamespace(models=types.SimpleNamespace())),
            "qdrant_client.http": types.SimpleNamespace(models=types.SimpleNamespace()),
            "qdrant_client.http.models": types.SimpleNamespace(BinaryQuantization=lambda *a, **k: None, BinaryQuantizationConfig=lambda *a, **k: None),
            "fastembed": types.SimpleNamespace(SparseTextEmbedding=lambda *a, **k: None, LateInteractionTextEmbedding=lambda *a, **k: None),
            "unstructured": types.SimpleNamespace(partition=types.SimpleNamespace(pdf=lambda *a, **k: []), documents=types.SimpleNamespace(elements=types.SimpleNamespace(Table=object, CompositeElement=object))),
            # "sentence_transformers": types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None),
            "transformers": types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(hidden_size=768))),
            "llama_index": types.SimpleNamespace(embeddings=types.SimpleNamespace(huggingface=types.SimpleNamespace(HuggingFaceEmbedding=lambda *a, **k: types.SimpleNamespace(_get_query_embedding=lambda t: [0.0], get_text_embedding_batch=lambda batch: [[0.0] * len(batch)])))),
            "ollama": types.SimpleNamespace(chat=lambda *a, **k: {"message": {"content": "dummy"}}, show=lambda *a, **k: {}),
        }
        # Patch sys.modules
        self.patch = mock.patch.dict(sys.modules, self.dummy_modules, clear=False)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_find_pathogen_keyword(self) -> None:
        from episcope.rag.retrieval.utils import find_pathogen_keyword

        self.assertEqual(find_pathogen_keyword("this_is_covid19_report.pdf"), "covid19")
        self.assertEqual(find_pathogen_keyword("/data/mpox/case.txt"), "mpox")
        self.assertIsNone(find_pathogen_keyword("random_document.pdf"))

    def test_token_count_and_budget(self) -> None:
        from episcope.rag.retrieval.utils import estimate_token_count, validate_token_budget, get_context_length

        messages = [
            {"role": "system", "content": "Hello world"},
            {"role": "user", "content": "Test message here"},
        ]
        est = estimate_token_count(messages)
        self.assertTrue(4 <= est <= 7)

        # Patch get_context_length to return a small context window
        with mock.patch(
            "episcope.rag.retrieval.utils.get_context_length", return_value=20
        ):
            # Should pass for short messages
            validate_token_budget("dummy", messages)
            # Should fail for long messages
            long_messages = [
                {"role": "user", "content": "X" * 1000},
            ]
            with self.assertRaises(ValueError):
                validate_token_budget("dummy", long_messages)


if __name__ == "__main__":
    unittest.main()