from __future__ import annotations

from copy import deepcopy

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import SearchResult
from episcope.workflows.precision_miner.config import PrecisionMinerConfig
from episcope.workflows.precision_miner.workflow import PrecisionMiner


def _result(*, id: str, text: str, score: float, section_type: str) -> SearchResult:
    return SearchResult(
        id=id,
        paper_id="paper-1",
        text=text,
        section_type=section_type,
        similarity_score=score,
        rank_score=score,
        source="test",
        artifacts={},
    )


class _StaticRetriever:
    def __init__(self, responses_by_query):
        self.responses_by_query = {query: list(results) for query, results in responses_by_query.items()}

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])]


class _UnusedGenerator(Generator):
    def generate(self, *args, **kwargs) -> Provenance:
        return Provenance(answer="{}", evidences=[])


def test_retrieve_chunks_filters_sections_and_keeps_best_duplicate() -> None:
    config = PrecisionMinerConfig(
        top_k=2,
        retrieval_templates=["q1", "q2"],
        section_filters=["Methods"],
    )
    retriever = _StaticRetriever(
        {
            "q1": [
                _result(id="1", text="same source mention", score=0.3, section_type="Methods"),
                _result(id="2", text="introduction mention", score=0.95, section_type="Introduction"),
            ],
            "q2": [
                _result(id="3", text="same source mention", score=0.8, section_type="Methods"),
                _result(id="4", text="second methods mention", score=0.6, section_type="Methods"),
            ],
        }
    )
    miner = PrecisionMiner(
        retriever=retriever,
        generator=_UnusedGenerator(),
        config=config,
    )

    chunks = miner.retrieve_chunks("paper-1")

    assert [chunk.text for chunk in chunks] == [
        "same source mention",
        "second methods mention",
    ]
    assert all(chunk.section_type == "Methods" for chunk in chunks)
    assert chunks[0].rank_score == 0.8
