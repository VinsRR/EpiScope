import json
import unittest
from unittest.mock import MagicMock

from episcope.db.in_memory_academic_db import InMemoryAcademicDB
from episcope.config.miner import FindDataSourcesConfig
from episcope.workflows.precision_miner_workflow import PrecisionMiner
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, ExtractionResult, ExtractionItem, SearchResult


class MockRetriever(AbstractRetriever):
    def retrieve(self, query: str, **kwargs):
        pass

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs) -> list[SearchResult]:
        return [
            SearchResult(
                id="1",
                paper_id=paper_id,
                text="The study used the NHANES dataset.",
                section_type="Methods",
                title="Methods",
                similarity_score=0.9,
                source="semantic",
            )
        ]


class MockGenerator(Generator):
    def generate(self, **kwargs) -> Provenance:
        extraction = ExtractionResult(
            description="The primary data source was the NHANES dataset.",
            items=[
                ExtractionItem(
                    item_type="data_source",
                    name="NHANES",
                    explanation="The study used the NHANES dataset.",
                    section_found="Methods",
                )
            ],
        )
        # Convert to dict, then to JSON string
        answer_json = json.dumps(extraction.__dict__, default=lambda o: o.__dict__)
        return Provenance(answer=answer_json, evidences=[])


class TestPrecisionMiner(unittest.TestCase):
    def setUp(self) -> None:
        self.db = InMemoryAcademicDB()
        self.retriever = MockRetriever()
        self.generator = MockGenerator()
        self.config = FindDataSourcesConfig()

    def test_run_find_data_sources(self):
        # Arrange
        paper_id = "test_paper"
        metadata = PaperMetadata(title="Test Paper", abstract="This is a test paper.")
        self.db.insert(paper_id, "metadata", "test_strategy", metadata.to_dict())

        miner = PrecisionMiner(
            retriever=self.retriever,
            generator=self.generator,
            config=self.config,
            academic_db=self.db
        )

        # Act
        result = miner.run(paper_id=paper_id)

        # Assert
        self.assertEqual(result.description, "The primary data source was the NHANES dataset.")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].name, "NHANES")


if __name__ == "__main__":
    unittest.main()