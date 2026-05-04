from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.ragas.models import RagPipelineConfig
from eval.ragas.testset_generation import (
    _build_query_distribution,
    _build_testset_models,
    _default_testset_max_tokens,
    _default_testset_reasoning_effort,
    _load_stored_chunk_documents,
    _limit_chunk_docs,
    _query_types_from_ratios,
    generate_explorer_testset,
)


class _SingleHopSpecificQuerySynthesizer:
    pass


class _MultiHopAbstractQuerySynthesizer:
    pass


class _MultiHopSpecificQuerySynthesizer:
    pass


class _FakeTestset:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_csv(self, path: str) -> None:
        self._frame.to_csv(path, index=False)

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeGenerator:
    last_kwargs = None

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

    def generate_with_chunks(
        self,
        *,
        chunks,
        testset_size,
        transforms_llm=None,
        query_distribution=None,
        **kwargs,
    ):
        type(self).last_kwargs = {
            "chunks": chunks,
            "testset_size": testset_size,
            "transforms_llm": transforms_llm,
            "query_distribution": query_distribution,
            **kwargs,
        }
        return _FakeTestset(
            pd.DataFrame(
                [
                    {
                        "user_input": "What data source was used?",
                        "reference": "NHANES was used.",
                        "reference_contexts": ["We used NHANES data."],
                        "persona_name": "reviewer",
                        "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
                    }
                ]
            )
        )


def test_build_query_distribution_reweights_default_buckets(monkeypatch) -> None:
    monkeypatch.setattr(
        "eval.ragas.testset_generation._require_default_query_distribution",
        lambda: lambda llm: [
            (_SingleHopSpecificQuerySynthesizer(), 0.5),
            (_MultiHopAbstractQuerySynthesizer(), 0.25),
            (_MultiHopSpecificQuerySynthesizer(), 0.25),
        ],
    )

    distribution = _build_query_distribution(
        object(),
        simple_ratio=0.2,
        reasoning_ratio=0.3,
        multi_context_ratio=0.5,
    )

    weights = [weight for _, weight in distribution]
    assert round(sum(weights), 6) == 1.0
    assert sorted(round(weight, 2) for weight in weights) == [0.2, 0.3, 0.5]


def test_query_types_follow_requested_ratios() -> None:
    assert _query_types_from_ratios(
        simple_ratio=1,
        reasoning_ratio=0,
        multi_context_ratio=0,
    ) == {"simple"}
    assert _query_types_from_ratios(
        simple_ratio=None,
        reasoning_ratio=None,
        multi_context_ratio=None,
    ) == {"simple", "reasoning", "multi_context"}


def test_gemini_testset_defaults_disable_thinking_and_raise_budget() -> None:
    assert _default_testset_max_tokens("gemini", "gemini-2.5-flash") == 8192
    assert _default_testset_reasoning_effort("google", "gemini-2.5-flash") == "none"
    assert _default_testset_reasoning_effort("google", "gemini-2.5-pro") is None
    assert _default_testset_max_tokens("openai", "gpt-4o-mini") is None


def test_build_testset_models_threads_generation_controls(monkeypatch) -> None:
    llm_configs = []

    def fake_build_llm(config):
        llm_configs.append(config)
        return f"llm-{len(llm_configs)}"

    monkeypatch.setattr(
        "eval.ragas.ragas_adapter._build_evaluator_llm",
        fake_build_llm,
    )
    monkeypatch.setattr(
        "eval.ragas.ragas_adapter._build_evaluator_embeddings",
        lambda config: "embeddings",
    )

    llm, embeddings, critic_llm = _build_testset_models(
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        critic_llm_provider="gemini",
        critic_llm_model="gemini-2.5-flash",
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
        generator_max_tokens=4096,
        generator_reasoning_effort="low",
    )

    assert (llm, embeddings, critic_llm) == ("llm-1", "embeddings", "llm-2")
    assert [config.max_tokens for config in llm_configs] == [4096, 4096]
    assert [config.reasoning_effort for config in llm_configs] == ["low", "low"]


def test_limit_chunk_docs_rejects_non_positive_limit() -> None:
    assert _limit_chunk_docs(["a", "b", "c"], 2) == ["a", "b"]

    try:
        _limit_chunk_docs(["a"], 0)
    except ValueError as exc:
        assert "max_chunks" in str(exc)
    else:
        raise AssertionError("Expected ValueError for max_chunks=0")


def test_generate_explorer_testset_writes_cases_and_threads_critic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "eval.ragas.testset_generation._require_ragas_testset",
        lambda: _FakeGenerator,
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._load_chunk_documents",
        lambda path, config: ["chunk-1", "chunk-2"],
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._build_testset_models",
        lambda **kwargs: ("generator-llm", "embeddings", "critic-llm"),
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._build_query_distribution",
        lambda *args, **kwargs: [("dist", 1.0)],
    )

    review_path = tmp_path / "generated_queries.jsonl"
    cases_path = tmp_path / "generated_cases.jsonl"
    csv_path = tmp_path / "generated_testset.csv"

    generated = generate_explorer_testset(
        path=tmp_path,
        pipeline_config=RagPipelineConfig(),
        llm_provider="google",
        llm_model="g-model",
        critic_llm_provider="google",
        critic_llm_model="c-model",
        embedding_provider="google",
        embedding_model="e-model",
        testset_size=1,
        out_review_jsonl=review_path,
        out_cases_jsonl=cases_path,
        out_csv=csv_path,
        simple_ratio=0.5,
        reasoning_ratio=0.25,
        multi_context_ratio=0.25,
    )

    assert review_path.exists()
    assert cases_path.exists()
    assert csv_path.exists()
    assert len(generated.qa_cases) == 1
    assert generated.qa_cases[0].user_input == "What data source was used?"
    assert _FakeGenerator.last_kwargs["transforms_llm"] == "critic-llm"
    assert _FakeGenerator.last_kwargs["query_distribution"] == [("dist", 1.0)]


def test_generate_explorer_testset_limits_chunks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "eval.ragas.testset_generation._require_ragas_testset",
        lambda: _FakeGenerator,
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._load_chunk_documents",
        lambda path, config: ["chunk-1", "chunk-2", "chunk-3"],
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._build_testset_models",
        lambda **kwargs: ("generator-llm", "embeddings", None),
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._build_query_distribution",
        lambda *args, **kwargs: [("dist", 1.0)],
    )

    generate_explorer_testset(
        path=tmp_path,
        pipeline_config=RagPipelineConfig(),
        llm_provider="google",
        llm_model="g-model",
        critic_llm_provider=None,
        critic_llm_model=None,
        embedding_provider="google",
        embedding_model="e-model",
        testset_size=1,
        out_review_jsonl=None,
        out_cases_jsonl=None,
        out_csv=None,
        max_chunks=2,
        simple_ratio=1,
        reasoning_ratio=0,
        multi_context_ratio=0,
    )

    assert _FakeGenerator.last_kwargs["chunks"] == ["chunk-1", "chunk-2"]


def test_load_stored_chunk_documents_uses_vectordb_namespace(monkeypatch) -> None:
    class _FakeVectorDb:
        def __init__(self):
            self.calls = []

        def get_points(self, namespace=None, filter=None):
            self.calls.append((namespace, filter))
            return [
                {
                    "id": "chunk-1",
                    "paper_id": "doc-123",
                    "text": "Stored chunk text.",
                    "section_title": "Methods",
                    "section_type": "Methods",
                    "is_metadata": False,
                }
            ]

    fake_db = _FakeVectorDb()
    monkeypatch.setattr(
        "eval.ragas.testset_generation._build_vector_db",
        lambda *args, **kwargs: fake_db,
    )
    monkeypatch.setattr(
        "eval.ragas.testset_generation._optional_langchain_document",
        lambda: None,
    )

    docs = _load_stored_chunk_documents("doc-123", RagPipelineConfig())

    assert docs == ["Stored chunk text."]
    assert fake_db.calls == [("doc-123", None)]
