"""Tests for the TextRetriever class.

These tests validate that the ``TextRetriever`` correctly delegates
query augmentation to HYDE when provided and returns contexts with
the expected fields.  A simple in‑memory ``FaissIndexer`` is used
for the index.
"""

# The "dummy" embedding model could be used by using unittest.mock to stub
# out the SentenceTransformer class: https://docs.python.org/3/library/unittest.mock.html

import unittest
from unittest.mock import Mock

from episcope.retrieve.text_retriever import TextRetriever
from episcope.index.faiss_indexer import FaissIndexer

from .configs import test_model_hf_embedding
embed_model = test_model_hf_embedding

class TestTextRetriever(unittest.TestCase):
    def test_retrieve_without_hyde(self) -> None:
        # Create an indexer and add a few documents
        indexer = FaissIndexer(
            embed_model=embed_model,
            batch_size=1)
        docs = [
            {"content": "Epidemiological studies show patterns."},
            {"content": "Literature reviews summarise prior work."},
        ]
        indexer.index_documents(docs)
        retriever = TextRetriever(indexer=indexer)
        contexts = retriever.retrieve("patterns", top_k=2)
        # Expect two contexts returned
        self.assertEqual(len(contexts), 2)
        # Ensure content field is present
        self.assertIn("content", contexts[0])

    def test_retrieve_with_hyde(self) -> None:
        # HYDE mock that returns a hypothetical document
        hyde = Mock()
        hyde.generate.return_value = "This is a hypothesis."
        indexer = FaissIndexer(
            embed_model=embed_model,
            batch_size=1
            )
        indexer.index_documents([{"content": "This is a document about data."}])
        retriever = TextRetriever(indexer=indexer, hyde=hyde)
        contexts = retriever.retrieve("data", top_k=1)
        # HYDE should have been called once
        hyde.generate.assert_called_once()
        self.assertEqual(len(contexts), 1)
        self.assertIn("content", contexts[0])


if __name__ == "__main__":
    unittest.main()