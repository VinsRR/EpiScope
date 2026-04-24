from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.ragas.io import load_simple_rag_qa_cases
from eval.ragas.models import RagPipelineConfig, SimpleRagQaCase
from eval.ragas.pipeline import run_case, run_cases


class _FakeEmbedder:
    model_name = "fake"

    def embed_text(self, text: str):
        text = text.lower()
        features = np.array(
            [
                text.count("health"),
                text.count("explore"),
                text.count("classify"),
                text.count("precision"),
                text.count("api"),
            ],
            dtype="float32",
        )
        norm = np.linalg.norm(features)
        if norm == 0:
            return features.tolist()
        return (features / norm).tolist()

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def test_load_simple_rag_qa_cases_resolves_relative_paths(tmp_path: Path) -> None:
    paper_path = tmp_path / "doc.txt"
    paper_path.write_text("Example content", encoding="utf-8")

    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "case_id": "case-1",
                "paper_path": "doc.txt",
                "user_input": "What is this?",
                "reference": "Example content",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_simple_rag_qa_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].paper_path == str(paper_path.resolve())


def test_run_case_on_local_text_file(monkeypatch, tmp_path: Path) -> None:
    from episcope.rag.embeddings.factory import EmbedderFactory

    monkeypatch.setattr(EmbedderFactory, "get_embedder", staticmethod(lambda model_name: _FakeEmbedder()))

    paper_path = tmp_path / "api_doc.txt"
    snippet = "The current API exposes GET /health, POST /explore, POST /classify, and POST /precision-miner."
    paper_path.write_text(snippet, encoding="utf-8")

    case = SimpleRagQaCase(
        schema_version="1",
        case_id="api-endpoints",
        paper_path=str(paper_path),
        user_input="What endpoints does the current API expose?",
        reference="The current API exposes GET /health, POST /explore, POST /classify, and POST /precision-miner.",
        reference_contexts=[snippet],
    )
    config = RagPipelineConfig(
        embed_model="fake",
        llm_provider="nollm",
        top_k=3,
    )

    result = run_case(case, config)

    assert result.run_error is None
    assert result.retrieval_count >= 1
    assert result.response is not None
    assert "GET /health" in result.response
    assert result.reference_context_recall == 1.0
    assert result.reference_context_precision is not None


def test_run_cases_captures_errors_when_continue_on_error(tmp_path: Path) -> None:
    missing_case = SimpleRagQaCase(
        schema_version="1",
        case_id="missing",
        paper_path=str(tmp_path / "missing.txt"),
        user_input="What is missing?",
        reference="Nothing",
    )
    config = RagPipelineConfig(continue_on_error=True)

    results = run_cases([missing_case], config)

    assert len(results) == 1
    assert results[0].run_error is not None
