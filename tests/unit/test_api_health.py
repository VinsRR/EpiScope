from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from episcope.api import app  # noqa: E402


def test_health_does_not_leak_mongo_uri(monkeypatch) -> None:
    # Even with a credential-bearing URI configured, /health must not expose it.
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://user:secret@cluster.example.net/")

    payload = TestClient(app).get("/health").json()

    mongo_uri = payload["defaults"]["mongo_uri"]
    assert "secret" not in str(mongo_uri)
    assert mongo_uri in (None, "***configured***")
    # The safe boolean is still reported.
    assert "mongo_uri_configured" in payload["checks"]


def test_health_exposes_kinds_catalog() -> None:
    payload = TestClient(app).get("/health").json()
    assert "kinds" in payload
    keys = {c["key"] for c in payload["kinds"]["classifiers"]}
    assert "data_accessibility" in keys


def test_health_reports_qdrant_url_null_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("qdrant_url", raising=False)
    payload = TestClient(app).get("/health").json()
    assert payload["defaults"]["qdrant_url"] is None


def test_backend_config_round_trips_local_fallback_paths() -> None:
    from pathlib import Path

    from episcope.api import BackendConfig

    config = BackendConfig(index_dir="/tmp/x-index", metadata_backup="/tmp/y-meta.json")
    runtime_config = config.to_runtime_config()

    assert runtime_config.index_dir == Path("/tmp/x-index")
    assert runtime_config.metadata_backup == Path("/tmp/y-meta.json")


def test_explore_endpoint_error_is_humanized(monkeypatch, tmp_path) -> None:
    # No QDRANT_URL/MONGO_URI configured, no local index built: /explore
    # raises a ValueError (400) that goes through humanize_error, same as
    # every other error path in this endpoint.
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("EPISCOPE_LOCAL_INDEX_DIR", str(tmp_path / "empty-index"))

    response = TestClient(app).post(
        "/explore", json={"query": "anything", "top_k": 1}
    )

    assert response.status_code == 400
    assert "dense embedding model" in response.json()["detail"]
