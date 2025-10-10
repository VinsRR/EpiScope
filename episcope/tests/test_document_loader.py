"""Tests for the document loader utilities.

These tests verify that the document loader can ingest plain text files
and assemble them into structured sections with metadata.  They also
exercise the factory for obtaining loaders by name.  For PDF files
Unstructured may not be installed, so the fallback behaviour is
implicitly tested when a ``.txt`` file is used instead of a real PDF.
"""

import os
import tempfile
import unittest
from pathlib import Path

from episcope.ingest.document_loader import (
    UnstructuredDocumentLoader,
    DocumentLoaderFactory,
)
from episcope.storage.in_memory_academic_db import InMemoryAcademicDB



class TestDocumentLoader(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary directory to hold test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        # Create two simple text files with paragraph breaks
        self.file1 = self.dir_path / "paper1.txt"
        self.file2 = self.dir_path / "paper2.md"
        self.paragraphs1 = [
            "First paragraph of the first paper.",
            "Second paragraph of the first paper.",
        ]
        self.paragraphs2 = [
            "Only one paragraph in the second paper.",
        ]
        self.file1.write_text("\n\n".join(self.paragraphs1), encoding="utf-8")
        self.file2.write_text("\n\n".join(self.paragraphs2), encoding="utf-8")

    def tearDown(self) -> None:
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_single_text_file(self) -> None:
        loader = UnstructuredDocumentLoader()
        sections, metadata = loader.load(self.file1)
        # Should create one section per paragraph
        self.assertEqual(len(sections), len(self.paragraphs1))
        # Verify that section content matches original paragraphs
        for sec, original in zip(sections, self.paragraphs1):
            self.assertEqual(sec.content.strip(), original)
        # Metadata title should be the filename stem
        self.assertEqual(metadata.title, self.file1.stem)

    def test_load_directory(self) -> None:
        loader = UnstructuredDocumentLoader()
        docs = loader.load_directory(self.dir_path)
        # There should be an entry per created file
        self.assertEqual(set(docs.keys()), {str(self.file1), str(self.file2)})
        # Validate contents of one of the loaded files
        sections1, meta1 = docs[str(self.file1)]
        self.assertEqual(len(sections1), len(self.paragraphs1))
        self.assertEqual(meta1.title, self.file1.stem)

    def test_factory_returns_loader(self) -> None:
        loader = DocumentLoaderFactory.get_loader("unstructured")
        self.assertIsInstance(loader, UnstructuredDocumentLoader)
        # Unknown loader should raise
        with self.assertRaises(ValueError):
            DocumentLoaderFactory.get_loader("unknown")

    def test_extract_paper_persists_data(self) -> None:
        loader = UnstructuredDocumentLoader()
        db = InMemoryAcademicDB()
        strategy_name = "test_strategy"

        loader.extract_paper(self.file1, strategy_name=strategy_name, db=db)

        # Verify metadata was inserted
        metadata = db.retrieve(self.file1.stem, "metadata", strategy_name)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.title, self.file1.stem)
        self.assertEqual(metadata.file_path, str(self.file1))

        # Verify sections were inserted
        sections = db.retrieve(self.file1.stem, "sections", strategy_name)
        self.assertIsNotNone(sections)
        self.assertEqual(len(sections), len(self.paragraphs1))
        for sec, original in zip(sections, self.paragraphs1):
            self.assertEqual(sec.content.strip(), original)



if __name__ == "__main__":  # pragma: no cover
    unittest.main()