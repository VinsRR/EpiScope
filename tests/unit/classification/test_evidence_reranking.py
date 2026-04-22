from __future__ import annotations

from copy import deepcopy

from episcope.schemas import SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.evidence_reranking import (
    GlobalCrossEncoderReranker,
    WithinLabelCrossEncoderReranker,
)


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


class _RecordingCrossEncoderReranker:
    def __init__(self, scores_by_query):
        self.scores_by_query = {query: dict(scores) for query, scores in scores_by_query.items()}
        self.calls = []

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.calls.append({"query": query, "texts": [result.text for result in results], "top_k": top_k})
        scores = self.scores_by_query.get(query, {})
        rescored = []
        for result in results:
            updated = deepcopy(result)
            updated.rank_score = scores.get(result.text, updated.rank_score)
            rescored.append(updated)
        rescored.sort(key=lambda item: item.rank_score, reverse=True)
        return rescored[:top_k]


def _config() -> BaseClassifierConfig:
    return BaseClassifierConfig(
        template_paragraphs={
            "open": ["long retrieval template for open"],
            "closed": ["long retrieval template for closed"],
        },
        cr_template_sentences={
            "open": ["condensed open evidence"],
            "closed": ["condensed closed evidence"],
        },
    )


def test_within_label_reranker_uses_condensed_templates() -> None:
    reranker = _RecordingCrossEncoderReranker(
        {
            "condensed open evidence": {"shared": 0.4, "open only": 0.95},
            "condensed closed evidence": {"shared": 0.2, "closed only": 0.9},
        }
    )
    strategy = WithinLabelCrossEncoderReranker(
        cross_encoder_reranker=reranker,
        config=_config(),
    )
    category_results = {
        "open": {
            "shared": _result(id="1", text="shared", score=0.8),
            "open only": _result(id="2", text="open only", score=0.6),
        },
        "closed": {
            "shared": _result(id="3", text="shared", score=0.7),
            "closed only": _result(id="4", text="closed only", score=0.5),
        },
    }

    reranked = strategy.rerank(category_results)

    assert [call["query"] for call in reranker.calls] == [
        "condensed open evidence",
        "condensed closed evidence",
    ]
    assert reranked["open"]["open only"].rank_score == 0.95
    assert reranked["open"]["open only"].artifacts["category"] == "open"
    assert reranked["closed"]["closed only"].rank_score == 0.9
    assert reranked["closed"]["closed only"].artifacts["category"] == "closed"


def test_global_reranker_builds_one_shared_candidate_pool() -> None:
    reranker = _RecordingCrossEncoderReranker(
        {
            "condensed open evidence": {"shared": 0.99, "open only": 0.7, "closed only": 0.3},
            "condensed closed evidence": {"shared": 0.1, "open only": 0.2, "closed only": 0.95},
        }
    )
    strategy = GlobalCrossEncoderReranker(
        cross_encoder_reranker=reranker,
        config=_config(),
    )
    category_results = {
        "open": {
            "shared": _result(id="1", text="shared", score=0.8),
            "open only": _result(id="2", text="open only", score=0.6),
        },
        "closed": {
            "shared": _result(id="3", text="shared", score=0.9),
            "closed only": _result(id="4", text="closed only", score=0.5),
        },
    }

    reranked = strategy.rerank(category_results)

    assert [call["query"] for call in reranker.calls] == [
        "condensed open evidence",
        "condensed closed evidence",
    ]
    assert set(reranker.calls[0]["texts"]) == {"shared", "open only", "closed only"}
    assert reranked["open"]["shared"].rank_score == 0.99
    assert reranked["closed"]["closed only"].rank_score == 0.95
