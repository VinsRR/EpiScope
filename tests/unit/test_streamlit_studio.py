from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from episcope.ui.api_client import StudioApiClient  # noqa: E402


APP_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "episcope"
    / "ui"
    / "streamlit_app.py"
)


@pytest.fixture()
def studio_client(monkeypatch):
    calls = []
    settings = {
        "llm_provider": "ollama",
        "llm_model": "qwen",
        "llm_temperature": 0.0,
        "workflow_top_k": 10,
        "evidence_reranker_kind": "none",
        "cross_encoder_model": None,
        "cross_encoder_top_k": 15,
        "loader": "unstructured",
        "chunker": "paragraph",
        "min_chunk_size": 20,
        "chunk_size": 600,
        "chunk_overlap": 100,
        "embed_model": "fake/embedder",
        "embed_provider": "auto",
        "index_backend": "file",
        "retrieval_mode": "dense_only",
    }
    workspace = {
        "id": "review",
        "name": "Review",
        "path": "/tmp/workspaces/review",
        "active": True,
        "indexed": True,
        "document_count": 1,
        "paper_count": 1,
        "task_count": 1,
        "settings": settings,
    }
    paper = {
        "paper_id": "paper-1",
        "title": "Paper One",
        "section_count": 1,
        "reference_count": 0,
        "source": "paper.txt",
        "metadata": {"title": "Paper One"},
    }
    document = {
        "id": "doc-1",
        "original_name": "paper.txt",
        "paper_id": "paper-1",
        "status": "indexed",
        "title": "Paper One",
        "section_count": 1,
        "updated_at": "2026-01-01T00:00:00Z",
    }
    task = {
        "key": "custom_task",
        "kind": "classifier",
        "label": "Custom task",
        "description": "",
        "top_k": 10,
        "labels": [{"code": "yes"}, {"code": "no"}],
        "multi_label": False,
        "default_label": "no",
        "retrieval_templates": [],
        "section_filters": None,
        "system_prompt": None,
        "user_prompt_template": None,
    }
    catalog = {
        "classifiers": [
            {"key": "data_accessibility", "label": "Data availability", "source": "builtin"},
            {"key": "custom_task", "label": "Custom task", "source": "workspace"},
        ],
        "miners": [
            {"key": "find_data_sources", "label": "Data sources", "source": "builtin"}
        ],
    }
    run_result = {
        "retrieval_count": 1,
        "retrieved_chunks": [
            {
                "paper_id": "paper-1",
                "section_type": "Methods",
                "text": "Evidence",
                "source": "paper.txt",
                "rank_score": 0.9,
            }
        ],
        "run_id": "run-1",
    }
    monkeypatch.setattr(
        StudioApiClient,
        "workspaces",
        lambda self: {"root": "/tmp/workspaces", "active_workspace_id": "review", "items": [workspace]},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "health",
        lambda self: {"checks": {"llm_provider": "ollama", "llm_api_key_configured": True}},
    )
    monkeypatch.setattr(StudioApiClient, "workspace", lambda self, _wid: workspace)
    monkeypatch.setattr(StudioApiClient, "documents", lambda self, _wid: [document])
    monkeypatch.setattr(StudioApiClient, "papers", lambda self, _wid: [paper])
    monkeypatch.setattr(
        StudioApiClient,
        "paper",
        lambda self, _wid, _pid: {
            "metadata": paper["metadata"],
            "sections": [{"title": "Methods", "content": "Evidence"}],
        },
    )
    monkeypatch.setattr(StudioApiClient, "tasks", lambda self, _wid: catalog)
    monkeypatch.setattr(StudioApiClient, "task", lambda self, _wid, _key: task)
    monkeypatch.setattr(
        StudioApiClient,
        "jobs",
        lambda self, _wid: [
            {"id": "job-running", "kind": "index", "status": "running", "current": 1, "total": 2, "message": "Working"},
            {"id": "job-interrupted", "kind": "classification", "status": "interrupted", "current": 0, "total": 1, "message": "Restarted"},
        ],
    )
    monkeypatch.setattr(
        StudioApiClient,
        "runs",
        lambda self, _wid: [
            {"id": "run-1", "created_at": "2026-01-01T00:00:00Z", "kind": "explore", "status": "completed", "summary": "query"}
        ],
    )
    monkeypatch.setattr(
        StudioApiClient,
        "run",
        lambda self, _wid, _rid: {"kind": "explore", "result": run_result},
    )
    monkeypatch.setattr(
        StudioApiClient, "download_run", lambda self, _wid, _rid, _fmt: b"paper_id,status\n"
    )

    def explore(self, workspace_id, request):
        calls.append(("explore", workspace_id, request))
        return run_result

    def workflow(self, workspace_id, **kwargs):
        calls.append(("workflow", workspace_id, kwargs))
        return {"id": "job-queued"}

    def validate_task(self, workspace_id, candidate):
        calls.append(("validate", workspace_id, candidate))
        return {"valid": True, "task": candidate}

    monkeypatch.setattr(StudioApiClient, "explore", explore)
    monkeypatch.setattr(StudioApiClient, "workflow", workflow)
    monkeypatch.setattr(StudioApiClient, "validate_task", validate_task)
    monkeypatch.setattr(
        StudioApiClient,
        "save_task",
        lambda self, workspace_id, candidate, editing_key=None: candidate,
    )
    monkeypatch.setattr(
        StudioApiClient,
        "update_settings",
        lambda self, workspace_id, changes: calls.append(("settings", workspace_id, changes)) or workspace,
    )
    monkeypatch.setattr(
        StudioApiClient,
        "upload_document",
        lambda self, workspace_id, **kwargs: calls.append(("upload", workspace_id, kwargs)) or {"id": "uploaded"},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "index",
        lambda self, workspace_id, document_ids, **kwargs: calls.append(("index", workspace_id, document_ids, kwargs)) or {"id": "index-job"},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "cancel_job",
        lambda self, workspace_id, job_id: calls.append(("cancel", workspace_id, job_id)) or {},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "retry_job",
        lambda self, workspace_id, job_id: calls.append(("retry", workspace_id, job_id)) or {},
    )
    return calls


def test_studio_home_renders_with_no_workspaces(monkeypatch) -> None:
    monkeypatch.setattr(
        StudioApiClient,
        "workspaces",
        lambda self: {"root": "/tmp/workspaces", "active_workspace_id": None, "items": []},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "health",
        lambda self: {
            "checks": {
                "llm_provider": "gemini",
                "llm_api_key_configured": False,
            }
        },
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    assert not app.exception
    assert any("EpiScope Studio" in title.value for title in app.title)


def page_app(monkeypatch, page: str) -> AppTest:
    monkeypatch.setenv("EPISCOPE_STUDIO_TEST_PAGE", page)
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["workspace_id"] = "review"
    return app.run(timeout=15)


@pytest.mark.parametrize(
    ("page", "title"),
    [
        ("library", "Library"),
        ("explore", "Explore"),
        ("workflows", "Workflows"),
        ("task_builder", "Task Builder"),
        ("jobs_results", "Jobs & Results"),
        ("settings", "Settings"),
    ],
)
def test_every_studio_page_renders(studio_client, monkeypatch, page, title) -> None:
    app = page_app(monkeypatch, page)
    assert not app.exception
    assert any(element.value == title for element in app.title)


def test_explore_and_workflow_forms_submit_to_api(studio_client, monkeypatch) -> None:
    app = page_app(monkeypatch, "explore")
    next(item for item in app.text_area if item.label == "Question or search query").input(
        "Where are the data?"
    )
    next(item for item in app.button if item.label == "Run").click()
    app.run(timeout=15)
    assert studio_client[0][0] == "explore"
    assert studio_client[0][2]["query"] == "Where are the data?"

    app = page_app(monkeypatch, "workflows")
    next(item for item in app.button if item.label == "Queue workflow").click()
    app.run(timeout=15)
    assert any(call[0] == "workflow" for call in studio_client)


def test_task_validation_and_settings_forms_submit(studio_client, monkeypatch) -> None:
    app = page_app(monkeypatch, "task_builder")
    next(item for item in app.button if item.label == "Validate").click()
    app.run(timeout=15)
    assert any(call[0] == "validate" for call in studio_client)

    app = page_app(monkeypatch, "settings")
    next(item for item in app.button if item.label == "Save runtime settings").click()
    app.run(timeout=15)
    assert any(call[0] == "settings" for call in studio_client)


def test_library_upload_index_and_job_controls(studio_client, monkeypatch) -> None:
    app = page_app(monkeypatch, "library")
    app.file_uploader[0].upload("new.txt", b"new paper", "text/plain")
    app.run(timeout=15)
    next(item for item in app.button if item.label == "Upload").click()
    app.run(timeout=15)
    assert any(call[0] == "upload" for call in studio_client)

    app = page_app(monkeypatch, "library")
    next(item for item in app.multiselect if item.label == "Documents to index").select(
        "doc-1"
    )
    app.run(timeout=15)
    next(item for item in app.button if item.label == "Start indexing").click()
    app.run(timeout=15)
    assert any(call[0] == "index" for call in studio_client)

    app = page_app(monkeypatch, "jobs_results")
    next(item for item in app.button if item.label == "Cancel").click()
    app.run(timeout=15)
    assert any(call[0] == "cancel" for call in studio_client)

    app = page_app(monkeypatch, "jobs_results")
    next(item for item in app.button if item.label == "Retry").click()
    app.run(timeout=15)
    assert any(call[0] == "retry" for call in studio_client)
