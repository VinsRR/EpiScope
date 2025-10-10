"""Tests for the PrecisionMinerRetriever.

These tests ensure that the PrecisionMinerRetriever can classify
papers and extract data source information using its heuristic
fallbacks.  The indexer used is a ``PaperIndexer`` with the dummy
embedding fallback.
"""

import unittest

from episcope.index.paper_indexer import PaperIndexer
from episcope.retrieve.precision import PrecisionMinerRetriever
from episcope.core.blueprints.data_blueprints import StructuredSection, PaperMetadata


class TestPrecisionMinerRetriever(unittest.TestCase):
    def test_classification_and_extraction(self) -> None:
        # Prepare a paper with a review‑like section and a data section
        sections = [
            StructuredSection(
                title="Background",
                content="In this review we summarise findings from numerous studies."
            ),
            StructuredSection(
                title="Results",
                content="The data shows an increase. Supplementary data can be found in the appendix."
            ),
        ]
        metadata = PaperMetadata(title="Sample Paper")
        indexer = PaperIndexer(min_chunk_size=5)
        indexer.index_paper(sections, metadata, paper_id="P1")
        retriever = PrecisionMinerRetriever(indexer=indexer)
        results = retriever.retrieve("What does the review say?", top_k=2, paper_ids=["P1"])
        # One result expected for one paper
        self.assertEqual(len(results), 1)
        res = results[0]
        # Should classify as literature_review because of the word 'review'
        self.assertEqual(res["classification"], "literature_review")
        # Should extract at least one data source from the second section
        self.assertTrue(len(res["data_sources"]) >= 1)
        # Data source explanation should mention 'Supplementary'
        self.assertIn("Supplementary", res["data_sources"][0]["explanation"])


if __name__ == "__main__":
    unittest.main()