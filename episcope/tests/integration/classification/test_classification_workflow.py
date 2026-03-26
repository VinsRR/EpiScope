from __future__ import annotations

from copy import deepcopy

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.config import DataAccessibilityClassifierConfig
from episcope.workflows.classification.schemas import DataAccessibility
from episcope.workflows.classification.workflow import PaperClassifier


def _result(*, id: str, text: str, score: float, category: str) -> SearchResult:
    return SearchResult(
        id=id,
        paper_id="paper-1",
        text=text,
        section_type="Methods",
        similarity_score=score,
        rank_score=score,
        source="test",
        artifacts={"category": category},
    )


class _StaticRetriever:
    def __init__(self, responses_by_query):
        self.responses_by_query = {query: list(results) for query, results in responses_by_query.items()}
        self.index_version = "test-index-v1"

    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])]


class _SequenceGenerator(Generator):
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.model_id = "fake-generator"

    def generate(self, *args, **kwargs) -> Provenance:
        self.calls.append({"args": args, "kwargs": kwargs})
        answer = self.answers.pop(0)
        return Provenance(answer=answer, evidences=[])


def _config() -> DataAccessibilityClassifierConfig:
    config = DataAccessibilityClassifierConfig(top_k=2, max_validation_retries=2)
    config.template_paragraphs = {
        "open": ["open-query"],
        "closed": ["closed-query"],
    }
    config.cr_template_sentences = {
        "open": ["open condensed"],
        "closed": ["closed condensed"],
    }
    return config


def test_classifier_run_detailed_returns_decision_trace_training_and_provenance() -> None:
    retriever = _StaticRetriever(
        {
            "open-query": [_result(id="1", text="Data are on Zenodo.", score=0.95, category="open")],
            "closed-query": [_result(id="2", text="Data cannot be shared.", score=0.2, category="closed")],
        }
    )
    generator = _SequenceGenerator(
        [
            """
            {
              "reasoning": "The evidence says the data are publicly available on Zenodo.",
              "confidence": 0.91,
              "class_probabilities": {"A": 0.91, "E": 0.09},
              "classification": ["A"]
            }
            """
        ]
    )
    metadata = PaperMetadata(title="Open dataset paper", abstract="This paper shares data.")
    classifier = PaperClassifier(retriever=retriever, generator=generator, config=_config())

    detailed = classifier.run_detailed("paper-1", metadata=metadata)

    assert detailed.decision.paper_id == "paper-1"
    assert detailed.decision.result.classification == [DataAccessibility.OPEN]
    assert [chunk.text for chunk in detailed.decision.top_evidence] == [
        "Data are on Zenodo.",
        "Data cannot be shared.",
    ]
    assert detailed.provenance.evidences[0].section == "open"
    assert detailed.provenance.evidences[0].index_version == "test-index-v1"
    assert detailed.provenance.evidences[0].model_id == "fake-generator"
    assert detailed.trace.raw_llm_response.strip().startswith("{")
    assert len(detailed.training.all_samples) == 1
    assert detailed.training.all_samples[0].parsed_ok is True


def test_classifier_run_falls_back_after_invalid_json_retries() -> None:
    retriever = _StaticRetriever(
        {
            "open-query": [_result(id="1", text="Ambiguous evidence.", score=0.7, category="open")],
            "closed-query": [],
        }
    )
    generator = _SequenceGenerator(["not json", "still not json"])
    metadata = PaperMetadata(title="Ambiguous paper", abstract="The evidence is unclear.")
    config = _config()
    classifier = PaperClassifier(retriever=retriever, generator=generator, config=config)

    decision = classifier.run("paper-1", metadata=metadata)

    assert decision.result.classification == config.default_classification
    assert decision.result.confidence == 0.0
    assert len(generator.calls) == config.max_validation_retries
