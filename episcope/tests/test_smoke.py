"""Basic import smoke tests for EpiScope.

These tests verify that core modules can be imported without
raising exceptions.  They do not require external services and use
the standard ``unittest`` framework rather than pytest.
"""

import unittest


class TestSmoke(unittest.TestCase):

    def test_import_indexers(self) -> None:
        from episcope.rag.indexing.faiss_indexer import FaissIndexer  # noqa: F401
        from episcope.rag.indexing.paper_indexer import PaperIndexer  # noqa: F401

    def test_import_retrievers(self) -> None:
        from episcope.rag.retrieval.text_retriever import TextRetriever  # noqa: F401
        from episcope.rag.retrieval.precision import PrecisionMinerRetriever  # noqa: F401

    def test_import_generator(self) -> None:
        from episcope.rag.generation.simple_generator import SimpleGenerator  # noqa: F401


if __name__ == "__main__":
    unittest.main()