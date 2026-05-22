from __future__ import annotations

from copy import deepcopy

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import SearchResult
from episcope.workflows.classification.config import DataAccessibilityClassifierConfig
from episcope.workflows.classification.evidence_reranking import GlobalCrossEncoderReranker
from episcope.workflows.classification.workflow import PaperClassifier


def _result(*, id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        id=id,
        paper_id="paper-1",
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

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])]


class _NoOpGenerator(Generator):
    def generate(self, *args, **kwargs) -> Provenance:
        return Provenance(
            answer="""
            {
              "reasoning": "The evidence favors openness.",
              "confidence": 0.75,
              "class_probabilities": {"A": 0.75, "E": 0.25},
              "classification": ["A"]
            }
            """,
            evidences=[],
        )


class _RecordingCrossEncoder:
    def __init__(self, scores_by_query):
        self.scores_by_query = {query: dict(scores) for query, scores in scores_by_query.items()}
        self.calls = []

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.calls.append({"query": query, "texts": [result.text for result in results]})
        scores = self.scores_by_query[query]
        rescored = []
        for result in results:
            updated = deepcopy(result)
            updated.rank_score = scores[result.text]
            rescored.append(updated)
        rescored.sort(key=lambda item: item.rank_score, reverse=True)
        return rescored[:top_k]


def test_classifier_uses_evidence_reranking_strategy_to_reorder_chunks() -> None:
    config = DataAccessibilityClassifierConfig(top_k=2)
    config.template_paragraphs = {
        "open": ["open-query"],
        "closed": ["closed-query"],
    }
    config.cr_template_sentences = {
        "open": ["open condensed"],
        "closed": ["closed condensed"],
    }
    retriever = _StaticRetriever(
        {
            "open-query": [
                _result(id="1", text="Shared evidence", score=0.4),
                _result(id="2", text="Open-specific evidence", score=0.3),
            ],
            "closed-query": [
                _result(id="3", text="Shared evidence", score=0.9),
                _result(id="4", text="Closed-specific evidence", score=0.2),
            ],
        }
    )
    cross_encoder = _RecordingCrossEncoder(
        {
            "open condensed": {
                "Shared evidence": 0.2,
                "Open-specific evidence": 0.95,
                "Closed-specific evidence": 0.1,
            },
            "closed condensed": {
                "Shared evidence": 0.99,
                "Open-specific evidence": 0.15,
                "Closed-specific evidence": 0.3,
            },
        }
    )
    classifier = PaperClassifier(
        retriever=retriever,
        generator=_NoOpGenerator(),
        config=config,
        evidence_reranker=GlobalCrossEncoderReranker(
            cross_encoder_reranker=cross_encoder,
            top_k=3,
        ),
    )

    chunks = classifier.get_relevant_chunks("paper-1", top_k=2)

    assert [call["query"] for call in cross_encoder.calls] == ["open condensed", "closed condensed"]
    assert [chunk.text for chunk in chunks] == ["Shared evidence", "Open-specific evidence"]
    assert [chunk.artifacts["category"] for chunk in chunks] == ["closed", "open"]
