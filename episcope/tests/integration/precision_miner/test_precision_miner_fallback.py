from __future__ import annotations

from copy import deepcopy

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.precision_miner.config import PrecisionMinerConfig
from episcope.workflows.precision_miner.workflow import PrecisionMiner


class _StaticRetriever:
    def __init__(self, responses_by_query):
        self.responses_by_query = {query: list(results) for query, results in responses_by_query.items()}

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])]


class _BadGenerator(Generator):
    def generate(self, *args, **kwargs) -> Provenance:
        return Provenance(answer="not valid json", evidences=[])


def test_precision_miner_falls_back_cleanly_on_invalid_generator_output() -> None:
    config = PrecisionMinerConfig(
        top_k=1,
        retrieval_templates=["source-query"],
        system_prompt="system",
        user_prompt_template="Chunks:\n{chunks_info}\nSchema:{schema}",
    )
    retriever = _StaticRetriever(
        {
            "source-query": [
                SearchResult(
                    id="1",
                    paper_id="paper-1",
                    text="We used NHANES data.",
                    section_type="Methods",
                    similarity_score=0.9,
                    rank_score=0.9,
                    source="test",
                    artifacts={},
                )
            ]
        }
    )
    miner = PrecisionMiner(retriever=retriever, generator=_BadGenerator(), config=config)
    metadata = PaperMetadata(title="Fallback paper", abstract="Bad JSON response.")

    detailed = miner.run_detailed("paper-1", metadata=metadata)

    assert detailed.result.items == []
    assert "Parsing/validation failed" in detailed.result.description
    assert detailed.trace.raw_llm_response == ""
    assert detailed.provenance.answer == ""
    assert detailed.provenance.evidences == []
