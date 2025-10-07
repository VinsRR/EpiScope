"""
Tests for the EmbeddingIndexer class.

These tests verify that chunking and sentence splitting work as
expected.  They avoid creating actual FAISS indexes or embedding
models by focusing on pure text operations.
"""
import unittest

# Stub external dependencies before imports


from episcope.parse.indexing.embeddings import EmbeddingIndexer
from episcope.parse.blueprints.data_blueprints import StructuredSection


class TestEmbeddingIndexer(unittest.TestCase):
    """Test cases for EmbeddingIndexer chunking utilities."""

    def setUp(self) -> None:
        # Patch the underlying embedder to avoid heavy dependencies.  The
        # PaperIndexer used by EmbeddingIndexer will fall back to a
        # length‑based embedding when the embedder is None, so we
        # simply instantiate the indexer with a small min_chunk_size.
        self.indexer = EmbeddingIndexer(
            model_name="dummy-model", chunking_strategy="paragraph", min_chunk_size=10
        )

    def tearDown(self) -> None:
        pass

    def test_split_into_sentences(self) -> None:
        text = "Sentence one. Sentence two! Sentence three?"
        sentences = self.indexer._split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Sentence one.")
        self.assertEqual(sentences[1], "Sentence two!")
        self.assertEqual(sentences[2], "Sentence three?")

    def test_paragraph_chunking_creates_chunks(self) -> None:
        content = "Paragraph one has enough words to be a chunk.\n\nShort.\n\nThis is another sufficiently long paragraph with more than five words."
        section = StructuredSection(title="Intro", content=content, section_type="Methods")
        chunks = self.indexer._paragraph_chunking(section)
        # Should create chunks only for paragraphs with >5 words
        self.assertEqual(len(chunks), 2)
        texts = [c['text'] for c in chunks]
        self.assertTrue(texts[0].startswith("Paragraph one"))
        self.assertTrue(texts[1].startswith("This is another"))


if __name__ == "__main__":
    unittest.main()