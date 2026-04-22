from __future__ import annotations

from copy import deepcopy

from episcope.schemas import SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.evidence_selection import ClassificationEvidenceSelector


def _result(*, id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        id=id,
        paper_id="p1",
        text=text,
        section_type="Methods",
        similarity_score=score,
        rank_score=score,
        source="test",
        artifacts={},
    )


class _StaticRetriever:
    def __init__(self, responses_by_query):
        self.responses_by_query = {query: list(results) for query, results in responses_by_query.items()}
        self.calls = []

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        self.calls.append({"query": query, "paper_id": paper_id, **kwargs})
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])]


def test_selector_deduplicates_by_text_and_keeps_best_scoring_category() -> None:
    config = BaseClassifierConfig(
        template_paragraphs={
            "open": ["open-q1", "open-q2"],
            "closed": ["closed-q1"],
        }
    )
    retriever = _StaticRetriever(
        {
            "open-q1": [
                _result(id="1", text="shared evidence", score=0.4),
                _result(id="2", text="open only", score=0.7),
            ],
            "open-q2": [
                _result(id="3", text="shared evidence", score=0.9),
            ],
            "closed-q1": [
                _result(id="4", text="shared evidence", score=0.6),
                _result(id="5", text="closed only", score=0.5),
            ],
        }
    )

    selector = ClassificationEvidenceSelector(retriever=retriever, config=config)
    selected = selector.select("p1", top_k=2)

    assert [chunk.text for chunk in selected] == ["shared evidence", "open only"]
    assert selected[0].artifacts["category"] == "open"
    assert selected[1].artifacts["category"] == "open"
    assert [call["query"] for call in retriever.calls] == ["open-q1", "open-q2", "closed-q1"]
