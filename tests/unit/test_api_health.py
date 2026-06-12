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
