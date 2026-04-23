from __future__ import annotations

import json

from typer.testing import CliRunner

from episcope.episcope import app
from episcope.rag.provenance import Provenance


runner = CliRunner()


class _FakeEmbedder:
    model_name = "fake-embedder/1"
    dim = 4

    def embed_text(self, text: str):
        size = float(len(text))
        lower = text.lower()
        return [
            1.0,
            float("zenodo" in lower),
            float("supplementary" in lower),
            size % 11.0,
        ]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


class _FakeClassifierGenerator:
    model_id = "fake-generator"

    def generate(self, contexts, **kwargs):
        return Provenance(
            answer="""
            {
              "reasoning": "The paper explicitly says the data are on Zenodo.",
              "confidence": 0.92,
              "class_probabilities": {"A": 0.92, "E": 0.08},
              "classification": ["A"]
            }
            """,
            evidences=[],
        )


def test_inspect_command_reads_text_file(tmp_path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(paper)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["paper_id"] == "paper"
    assert payload["section_count"] >= 1
    assert payload["reference_count"] == 0


def test_index_and_papers_use_local_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name: _FakeEmbedder(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    db_backup = tmp_path / "academic_db.json"

    index_result = runner.invoke(
        app,
        [
            "index",
            str(paper),
            "--index-dir",
            str(index_dir),
            "--db-backup",
            str(db_backup),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert index_result.exit_code == 0
    index_payload = json.loads(index_result.stdout)
    assert index_payload["status"] == "ok"
    assert index_payload["papers"][0]["paper_id"] == "paper"

    papers_result = runner.invoke(
        app,
        [
            "papers",
            "--db-backup",
            str(db_backup),
        ],
    )

    assert papers_result.exit_code == 0
    papers_payload = json.loads(papers_result.stdout)
    assert papers_payload["paper_ids"] == ["paper"]


def test_explore_path_runs_without_llm_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name: _FakeEmbedder(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "explore",
            "Zenodo",
            "--path",
            str(paper),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] is None
    assert payload["retrieval_count"] >= 1
    assert any("Zenodo" in chunk["text"] for chunk in payload["retrieved_chunks"])


def test_classify_file_uses_transient_local_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeClassifierGenerator(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "classify",
            "--file",
            str(paper),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["paper_id"] == "paper"
    assert payload["result"]["classification"] == ["open"]
