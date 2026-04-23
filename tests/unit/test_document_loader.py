from __future__ import annotations

import sys
import types

from episcope.rag.ingestion.document_loader import GrobidDocumentLoader


def test_grobid_loader_uses_env_url(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, grobid_server: str) -> None:
            captured["grobid_server"] = grobid_server

    monkeypatch.setenv("GROBID_URL", "http://grobid:8070")
    monkeypatch.setitem(
        sys.modules,
        "episcope.rag.ingestion.local_grobid_client",
        types.SimpleNamespace(GrobidClient=FakeClient),
    )

    loader = GrobidDocumentLoader()

    assert loader.grobid_url == "http://grobid:8070"
    assert captured["grobid_server"] == "http://grobid:8070"


def test_grobid_loader_explicit_url_overrides_env(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, grobid_server: str) -> None:
            captured["grobid_server"] = grobid_server

    monkeypatch.setenv("GROBID_URL", "http://grobid:8070")
    monkeypatch.setitem(
        sys.modules,
        "episcope.rag.ingestion.local_grobid_client",
        types.SimpleNamespace(GrobidClient=FakeClient),
    )

    loader = GrobidDocumentLoader(grobid_url="http://custom-grobid:8070")

    assert loader.grobid_url == "http://custom-grobid:8070"
    assert captured["grobid_server"] == "http://custom-grobid:8070"
