"""
Unit tests for the RAG factory.

These tests ensure that known methods are created and unknown
identifiers raise an error.  The concrete classes themselves are
otherwise not exercised.
"""
import unittest
from unittest import mock


class TestRAGFactory(unittest.TestCase):
    def test_known_method(self) -> None:
        import sys
        import types
        # Stub external dependencies to allow import.  We provide simple
        # placeholders for heavy modules that may not be available.
        dummy_modules = {
            "qdrant_client": types.SimpleNamespace(http=types.SimpleNamespace(models=types.SimpleNamespace())),
            "qdrant_client.http": types.SimpleNamespace(models=types.SimpleNamespace()),
            "qdrant_client.http.models": types.SimpleNamespace(
                BinaryQuantization=lambda *a, **k: None,
                BinaryQuantizationConfig=lambda *a, **k: None,
            ),
            "fastembed": types.SimpleNamespace(
                SparseTextEmbedding=lambda *a, **k: None,
                LateInteractionTextEmbedding=lambda *a, **k: None,
            ),
            "unstructured": types.SimpleNamespace(
                partition=types.SimpleNamespace(pdf=lambda *a, **k: []),
                documents=types.SimpleNamespace(elements=types.SimpleNamespace(Table=object, CompositeElement=object)),
            ),
            "unstructured.partition": types.SimpleNamespace(pdf=lambda *a, **k: []),
            "unstructured.partition.pdf": types.SimpleNamespace(partition_pdf=lambda *a, **k: []),
            "transformers": types.SimpleNamespace(
                AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(hidden_size=768))
            ),
            "llama_index": types.SimpleNamespace(
                embeddings=types.SimpleNamespace(
                    huggingface=types.SimpleNamespace(
                        HuggingFaceEmbedding=lambda *a, **k: types.SimpleNamespace(
                            _get_query_embedding=lambda t: [0.0],
                            get_text_embedding_batch=lambda batch: [[0.0] * len(batch)],
                        )
                    )
                )
            ),
            "ollama": types.SimpleNamespace(chat=lambda *a, **k: None, Client=lambda *a, **k: types.SimpleNamespace(chat=lambda *a, **k: None)),
            "tqdm": types.SimpleNamespace(tqdm=lambda x, *a, **k: x),
        }
        with mock.patch.dict(sys.modules, dummy_modules, clear=False):
            from episcope.rag.retrieval.rag.factory import RAGFactory  # type: ignore
            klass = RAGFactory._registry.get("text")  # type: ignore[attr-defined]
            self.assertIsNotNone(klass)
            self.assertEqual(klass.__name__, "TextRAG")

    def test_unknown_method(self) -> None:
        import sys
        import types
        dummy_modules = {
            "qdrant_client": types.SimpleNamespace(http=types.SimpleNamespace(models=types.SimpleNamespace())),
            "qdrant_client.http": types.SimpleNamespace(models=types.SimpleNamespace()),
            "qdrant_client.http.models": types.SimpleNamespace(
                BinaryQuantization=lambda *a, **k: None,
                BinaryQuantizationConfig=lambda *a, **k: None,
            ),
            "fastembed": types.SimpleNamespace(
                SparseTextEmbedding=lambda *a, **k: None,
                LateInteractionTextEmbedding=lambda *a, **k: None,
            ),
            "unstructured": types.SimpleNamespace(
                partition=types.SimpleNamespace(pdf=lambda *a, **k: []),
                documents=types.SimpleNamespace(elements=types.SimpleNamespace(Table=object, CompositeElement=object)),
            ),
            "unstructured.partition": types.SimpleNamespace(pdf=lambda *a, **k: []),
            "unstructured.partition.pdf": types.SimpleNamespace(partition_pdf=lambda *a, **k: []),
            "transformers": types.SimpleNamespace(
                AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(hidden_size=768))
            ),
            "llama_index": types.SimpleNamespace(
                embeddings=types.SimpleNamespace(
                    huggingface=types.SimpleNamespace(
                        HuggingFaceEmbedding=lambda *a, **k: types.SimpleNamespace(
                            _get_query_embedding=lambda t: [0.0],
                            get_text_embedding_batch=lambda batch: [[0.0] * len(batch)],
                        )
                    )
                )
            ),
            "ollama": types.SimpleNamespace(chat=lambda *a, **k: None, Client=lambda *a, **k: types.SimpleNamespace(chat=lambda *a, **k: None)),
            "tqdm": types.SimpleNamespace(tqdm=lambda x, *a, **k: x),
        }
        with mock.patch.dict(sys.modules, dummy_modules, clear=False):
            from episcope.rag.retrieval.rag.factory import RAGFactory  # type: ignore
            with self.assertRaises(ValueError):
                RAGFactory.get("unknown")


if __name__ == "__main__":
    unittest.main()