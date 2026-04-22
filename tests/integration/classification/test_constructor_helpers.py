from __future__ import annotations

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.workflows.classification.config import DataAccessibilityClassifierConfig
from episcope.workflows.classification.evidence_reranking import (
    GlobalCrossEncoderReranker,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.classification.workflow import PaperClassifier


class _StubRetriever:
    def retrieve_by_paper(self, query: str, paper_id: str, **kwargs):
        return []


class _StubGenerator(Generator):
    def generate(self, *args, **kwargs) -> Provenance:
        return Provenance(answer="{}", evidences=[])


class _SentinelStrategy:
    def configure(self, config):
        return self


def test_with_global_cross_encoder_uses_factory_reranker(monkeypatch) -> None:
    sentinel = _SentinelStrategy()

    def _fake_factory(cls, **kwargs):
        assert kwargs["model_name"] == "fake-model"
        assert kwargs["top_k"] == 7
        return sentinel

    monkeypatch.setattr(GlobalCrossEncoderReranker, "from_huggingface", classmethod(_fake_factory))
    config = DataAccessibilityClassifierConfig()

    classifier = PaperClassifier.with_global_cross_encoder(
        retriever=_StubRetriever(),
        generator=_StubGenerator(),
        model_name="fake-model",
        config=config,
        top_k=7,
    )

    assert classifier.config is config
    assert classifier.evidence_reranker is sentinel


def test_with_within_label_cross_encoder_uses_factory_reranker(monkeypatch) -> None:
    sentinel = _SentinelStrategy()

    def _fake_factory(cls, **kwargs):
        assert kwargs["model_name"] == "fake-model"
        assert kwargs["top_k"] == 3
        return sentinel

    monkeypatch.setattr(WithinLabelCrossEncoderReranker, "from_huggingface", classmethod(_fake_factory))
    config = DataAccessibilityClassifierConfig()

    classifier = PaperClassifier.with_within_label_cross_encoder(
        retriever=_StubRetriever(),
        generator=_StubGenerator(),
        model_name="fake-model",
        config=config,
        top_k=3,
    )

    assert classifier.config is config
    assert classifier.evidence_reranker is sentinel
