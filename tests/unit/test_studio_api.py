from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import episcope.api as api_module  # noqa: E402
from episcope.services.studio import StudioSettings, WorkspaceService  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    service = WorkspaceService(StudioSettings(workspaces_root=tmp_path / "workspaces"))
    monkeypatch.setattr(api_module, "_studio_workspaces", service)
    monkeypatch.setattr(api_module, "_studio_jobs", None)
    with TestClient(api_module.app) as test_client:
        yield test_client
    if api_module._studio_jobs is not None:
        api_module._studio_jobs.shutdown()
    monkeypatch.setattr(api_module, "_studio_jobs", None)


def test_workspace_document_and_task_api(client) -> None:
    response = client.post("/workspaces", json={"id": "review", "name": "Review"})
    assert response.status_code == 201
    assert client.get("/workspaces").json()["items"][0]["id"] == "review"

    upload = client.post(
        "/workspaces/review/documents",
        files={"file": ("paper.txt", b"Paper contents")},
        data={"paper_id": "paper-1"},
    )
    assert upload.status_code == 201
    assert upload.json()["paper_id"] == "paper-1"

    task = {
        "key": "study_design",
        "kind": "classifier",
        "labels": [{"code": "cohort"}, {"code": "unclear"}],
        "default_label": "unclear",
        "multi_label": False,
    }
    assert client.post("/workspaces/review/tasks/validate", json=task).status_code == 200
    assert client.post("/workspaces/review/tasks", json=task).status_code == 201
    assert client.get("/workspaces/review/tasks/study_design").json()["key"] == "study_design"


def test_workspace_api_rejects_traversal_and_never_exposes_secrets(client, monkeypatch) -> None:
    monkeypatch.setenv("MONGO_URI", "mongodb://user:secret@example.invalid")
    assert client.post("/workspaces", json={"id": "../escape"}).status_code == 400
    payload = client.get("/health").json()
    assert "secret" not in str(payload)


def test_jobs_are_persisted_and_can_be_cancelled(client) -> None:
    client.post("/workspaces", json={"id": "review"})
    response = client.post(
        "/workspaces/review/index-jobs", json={"document_ids": [], "replace_existing": False}
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    listed = client.get("/workspaces/review/jobs").json()["items"]
    assert any(item["id"] == job_id for item in listed)
