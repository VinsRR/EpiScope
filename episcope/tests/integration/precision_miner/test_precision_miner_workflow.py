from __future__ import annotations

from copy import deepcopy

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.precision_miner.config import PrecisionMinerConfig
from episcope.workflows.precision_miner.workflow import PrecisionMiner


def _result(*, id: str, text: str, score: float, section_type: str = "Methods") -> SearchResult:
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


class _SequenceGenerator(Generator):
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, *args, **kwargs) -> Provenance:
        self.calls.append({"args": args, "kwargs": kwargs})
        return Provenance(answer=self.answers.pop(0), evidences=[])


def test_precision_miner_run_detailed_returns_result_trace_and_chunks() -> None:
    config = PrecisionMinerConfig(
        top_k=2,
        retrieval_templates=["source-query"],
        section_filters=["Methods"],
        system_prompt="You find data sources.",
        user_prompt_template="Title: {title}\nChunks:\n{chunks_info}\nSchema: {schema}",
    )
    retriever = _StaticRetriever(
        {
            "source-query": [
                _result(id="1", text="We used NHANES data.", score=0.9),
                _result(id="2", text="Introduction text.", score=0.95, section_type="Introduction"),
            ]
        }
    )
    generator = _SequenceGenerator(
        [
            """
            {
              "description": "One source identified.",
              "items": [
                {
                  "name": "NHANES",
                  "url": null,
                  "explanation": "The study explicitly says it used NHANES.",
                  "raw_text": "We used NHANES data."
                }
              ]
            }
            """
        ]
    )
    metadata = PaperMetadata(title="NHANES study", abstract="Uses survey data.")
    miner = PrecisionMiner(retriever=retriever, generator=generator, config=config)

    detailed = miner.run_detailed("paper-1", metadata=metadata)

    assert detailed.paper_id == "paper-1"
    assert detailed.result.description == "One source identified."
    assert [item.name for item in detailed.result.items] == ["NHANES"]
    assert [chunk.text for chunk in detailed.relevant_chunks] == ["We used NHANES data."]
    assert detailed.trace.raw_llm_response.strip().startswith("{")
    assert len(generator.calls) == 1
