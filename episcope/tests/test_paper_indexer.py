"""Unit tests for the PaperIndexer class.

These tests exercise the basic indexing and search behaviour of
``PaperIndexer``.  They avoid external dependencies by relying on
the fallback length‑based embedding if transformer models are not
available.  The focus is on verifying that chunks are created
appropriately and that the search returns the expected contexts.
"""

import unittest
from episcope.index.paper_indexer import PaperIndexer
from episcope.parse.blueprints.data_blueprints import StructuredSection, PaperMetadata


class TestPaperIndexer(unittest.TestCase):
    def test_index_and_search_paper(self) -> None:
        # Create a paper with two sections; one long paragraph and one short
        sections = [
            StructuredSection(
                title="Introduction",
                content="This introduction contains enough words to be considered for indexing. It should be included as a chunk.",
                section_type="Intro",
            ),
            StructuredSection(
                title="Methods",
                content="Short.",
                section_type="Methods",
            ),
        ]
        metadata = PaperMetadata(title="Test Paper", abstract="An abstract with content.")
        indexer = PaperIndexer(min_chunk_size=10, embed_model="all-MiniLM-L6-v2")
        indexer.index_paper(sections, metadata, paper_id="paperX")
        # Ensure embeddings and metadata were stored
        self.assertIn("paperX", indexer._embeddings)
        self.assertEqual(len(indexer._metadata["paperX"]), 2)  # abstract + one paragraph
        # Search for a keyword in the introduction
        results = indexer.search("introduction", top_k=3)

        # Should return at least one result from paperX
        self.assertTrue(any(r["paper_id"] == "paperX" for r in results))
        # The content of the first result should include the introduction text
        first_content = results[0]["content"]
        self.assertIn("introduction contains", first_content)


if __name__ == "__main__":
    unittest.main()