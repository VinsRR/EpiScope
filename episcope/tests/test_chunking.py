"""
Tests for the chunking utilities.
"""
import unittest

from episcope.parse.indexing.chunking import paragraph_chunking, split_into_sentences
from episcope.parse.blueprints.data_blueprints import StructuredSection

class TestChunking(unittest.TestCase):
    """Test cases for chunking utilities."""

    def test_split_into_sentences(self) -> None:
        text = "Sentence one. Sentence two! Sentence three?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Sentence one.")
        self.assertEqual(sentences[1], "Sentence two!")
        self.assertEqual(sentences[2], "Sentence three?")

    def test_paragraph_chunking_creates_chunks(self) -> None:
        content = "Paragraph one has enough words to be a chunk.\n\nShort.\n\nThis is another sufficiently long paragraph with more than five words."
        section = StructuredSection(title="Intro", content=content, section_type="Methods")
        chunks = paragraph_chunking(section, min_chunk_size=10)
        # Should create chunks only for paragraphs with >5 words
        self.assertEqual(len(chunks), 2)
        texts = [c['text'] for c in chunks]
        self.assertTrue(texts[0].startswith("Paragraph one"))
        self.assertTrue(texts[1].startswith("This is another"))

if __name__ == "__main__":
    unittest.main()

