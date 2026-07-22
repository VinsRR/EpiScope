from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from episcope.episcope import app
from episcope.schemas import PaperMetadata, StructuredSection
from episcope.services.studio import (
    CorpusService,
    StudioRepository,
    StudioSettings,
    TaskService,
    WorkspaceService,
)


class _FakeEmbedder:
    model_name = "fake/embedder"
    dim = 4

    def embed_texts(self, texts):
        return [[1.0, float(index), 0.0, 0.0] for index, _ in enumerate(texts)]


class _FakeLoader:
    def load(self, path):
        text = Path(path).read_text(encoding="utf-8")
        return (
            [StructuredSection(title="Methods", content=text, section_type="Methods")],
            PaperMetadata(title="A paper", abstract="A sufficiently useful abstract."),
            [],
        )


@pytest.fixture()
def workspace_service(tmp_path) -> WorkspaceService:
    return WorkspaceService(StudioSettings(workspaces_root=tmp_path / "workspaces"))


def test_workspace_managed_root_and_settings_round_trip(workspace_service) -> None:
    created = workspace_service.create("review-one", name="Review One")
    assert created["id"] == "review-one"
    assert workspace_service.list()[0]["name"] == "Review One"

    updated = workspace_service.update_settings(
        "review-one",
        {"llm_provider": "ollama", "llm_model": "qwen", "workflow_top_k": 17},
    )
    assert updated["settings"]["llm_provider"] == "ollama"
    assert updated["settings"]["workflow_top_k"] == 17
    assert workspace_service.get("review-one").llm_model == "qwen"


@pytest.mark.parametrize("workspace_id", ["../escape", "/absolute", "", "a/b"])
def test_workspace_ids_cannot_escape_root(workspace_service, workspace_id) -> None:
    with pytest.raises(ValueError):
        workspace_service.workspace_path(workspace_id)


def test_upload_validation_deduplication_and_paper_id_collision(workspace_service) -> None:
    workspace_service.create("review")
    corpus = CorpusService(workspace_service.get("review"), upload_limit_mb=1)
    first = corpus.upload("paper.txt", b"A useful paper body.", paper_id="paper-1")
    duplicate = corpus.upload("paper-copy.txt", b"A useful paper body.", paper_id="other")
    assert duplicate["id"] == first["id"]
    assert duplicate["duplicate"] is True

    with pytest.raises(ValueError, match="already assigned"):
        corpus.upload("different.txt", b"Different body.", paper_id="paper-1")
    with pytest.raises(ValueError, match="Unsupported"):
        corpus.upload("paper.exe", b"data", paper_id="bad")
    with pytest.raises(ValueError, match="path"):
        corpus.upload("../paper.txt", b"data", paper_id="bad")


def test_task_service_is_workspace_scoped_and_validates_prompts(workspace_service) -> None:
    workspace_service.create("one")
    workspace_service.create("two")
    task = {
        "key": "study_design",
        "kind": "classifier",
        "labels": [{"code": "cohort"}, {"code": "unclear"}],
        "default_label": "unclear",
        "multi_label": False,
    }
    service = TaskService(workspace_service.get("one"))
    service.save(task)
    assert service.get("study_design").key == "study_design"
    assert TaskService(workspace_service.get("two")).list()["classifiers"]
    with pytest.raises(KeyError):
        TaskService(workspace_service.get("two")).get("study_design")

    invalid = {**task, "key": "bad_prompt", "user_prompt_template": "Bad {missing}"}
    with pytest.raises(ValueError, match="unrecognised placeholder"):
        service.validate(invalid)


def test_repository_marks_active_jobs_interrupted_only_when_requested(workspace_service) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    repository = StudioRepository(workspace)
    job = repository.create_job("index", {"document_ids": []}, total=0)
    assert StudioRepository(workspace).get_job(job["id"])["status"] == "queued"
    repository.interrupt_active_jobs()
    assert repository.get_job(job["id"])["status"] == "interrupted"


def test_local_indexing_persists_paper_and_rolls_back_on_failure(
    workspace_service, monkeypatch
) -> None:
    workspace_service.create("review")
    workspace = workspace_service.get("review")
    corpus = CorpusService(workspace)
    document = corpus.upload(
        "paper.txt",
        b"This methods paragraph contains enough words to become one useful indexed chunk.",
        paper_id="paper-1",
    )
    monkeypatch.setattr(
        "episcope.services.studio.DocumentLoaderFactory.get_loader",
        lambda *_args, **_kwargs: _FakeLoader(),
    )
    monkeypatch.setattr(
        "episcope.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: _FakeEmbedder(),
    )
    result = corpus.index_documents(
        [document["id"]],
        replace_existing=False,
        job_id="job-one",
        progress=lambda *_: None,
        cancelled=lambda: False,
    )
    assert result["errors"] == []
    assert corpus.list_papers()[0]["paper_id"] == "paper-1"
    cli_result = CliRunner().invoke(
        app,
        ["papers", "--workspace", str(workspace.root), "--format", "json"],
    )
    assert cli_result.exit_code == 0
    assert json.loads(cli_result.stdout)["paper_ids"] == ["paper-1"]
    index_metadata = workspace.resolve_path(workspace.index_dir) / "metadata.json"
    before = index_metadata.read_bytes()

    monkeypatch.setattr(
        "episcope.services.studio.EmbedderFactory.get_embedder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("embedder failed")),
    )
    with pytest.raises(RuntimeError, match="embedder failed"):
        corpus.index_documents(
            [document["id"]],
            replace_existing=True,
            job_id="job-two",
            progress=lambda *_: None,
            cancelled=lambda: False,
        )
    assert index_metadata.read_bytes() == before
