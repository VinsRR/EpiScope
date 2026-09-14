from __future__ import annotations

import pytest

from epilens.clients import AnthropicClient
from epilens.db import InMemoryAcademicDB
from epilens.rag.retrieval.candidates import SemanticCandidateRetriever
from epilens.services.runtime import (
    EpiLensRuntime,
    RuntimeConfig,
    _coerce_llm_provider,
)


def test_build_llm_client_dispatches_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    runtime = EpiLensRuntime(RuntimeConfig(llm_provider="anthropic"))
    assert isinstance(runtime.build_llm_client(), AnthropicClient)


def test_coerce_llm_provider_accepts_anthropic() -> None:
    assert _coerce_llm_provider("Anthropic") == "anthropic"


def test_coerce_llm_provider_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        _coerce_llm_provider("bogus")


# ---------------------------------------------------------------------------
# RuntimeConfig.from_settings(): qdrant_url reads the raw env var
# ---------------------------------------------------------------------------
def test_from_settings_qdrant_url_is_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("qdrant_url", raising=False)
    config = RuntimeConfig.from_settings()
    assert config.qdrant_url is None


def test_from_settings_qdrant_url_reads_env_var(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    config = RuntimeConfig.from_settings()
    assert config.qdrant_url == "http://example:6333"


def test_from_settings_reads_local_fallback_paths(monkeypatch, tmp_path) -> None:
    index_dir = tmp_path / "some-index"
    metadata_backup = tmp_path / "some-meta.json"
    monkeypatch.setenv("EPILENS_LOCAL_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("EPILENS_LOCAL_METADATA_BACKUP", str(metadata_backup))
    config = RuntimeConfig.from_settings()
    assert config.index_dir == index_dir
    assert config.metadata_backup == metadata_backup


# ---------------------------------------------------------------------------
# build_db(): local fallback vs. strict Mongo-required path
# ---------------------------------------------------------------------------
def test_build_db_falls_back_to_in_memory_when_unconfigured(tmp_path) -> None:
    backup = tmp_path / "db.json"
    runtime = EpiLensRuntime(RuntimeConfig(mongo_uri=None, metadata_backup=backup))
    db = runtime.build_db()
    assert isinstance(db, InMemoryAcademicDB)


def test_build_db_still_constructs_mongo_when_configured(monkeypatch) -> None:
    captured = {}

    class _FakeMongoAcademicDB:
        def __init__(self, uri, db_name):
            captured["uri"] = uri
            captured["db_name"] = db_name

    monkeypatch.setattr(
        "epilens.services.runtime.MongoAcademicDB", _FakeMongoAcademicDB
    )
    runtime = EpiLensRuntime(
        RuntimeConfig(mongo_uri="mongodb://example:27017", mongo_db_name="mydb")
    )
    db = runtime.build_db()
    assert isinstance(db, _FakeMongoAcademicDB)
    assert captured == {"uri": "mongodb://example:27017", "db_name": "mydb"}


# ---------------------------------------------------------------------------
# build_retriever(): local fallback vs. strict Qdrant-required path
# ---------------------------------------------------------------------------
def test_build_retriever_falls_back_to_file_db_when_unconfigured(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class _FakeFileDB:
        def __init__(self, index_dir):
            captured["index_dir"] = index_dir

        def get_embedding_model(self):
            return "fake-model"

        def capabilities(self):
            return {}

    monkeypatch.setattr("epilens.services.runtime.FileDB", _FakeFileDB)
    monkeypatch.setattr(
        "epilens.rag.embeddings.factory.EmbedderFactory.get_embedder",
        staticmethod(lambda model_name, **_: object()),
    )
    index_dir = tmp_path / "myindex"
    runtime = EpiLensRuntime(RuntimeConfig(qdrant_url=None, index_dir=index_dir))

    retriever = runtime.build_retriever()

    assert captured["index_dir"] == str(index_dir)
    assert retriever.use_rerank is False
    assert len(retriever.candidate_retrievers) == 1
    assert isinstance(retriever.candidate_retrievers[0], SemanticCandidateRetriever)


def test_build_retriever_still_constructs_qdrant_when_configured(monkeypatch) -> None:
    captured = {}

    class _FakeQdrantDB:
        def __init__(self, collection, url):
            captured["collection"] = collection
            captured["url"] = url

        def get_embedding_model(self):
            return "fake-model"

        def capabilities(self):
            return {}

    monkeypatch.setattr("epilens.services.runtime.QdrantDB", _FakeQdrantDB)
    monkeypatch.setattr(
        "epilens.rag.embeddings.factory.EmbedderFactory.get_embedder",
        staticmethod(lambda model_name, **_: object()),
    )
    runtime = EpiLensRuntime(
        RuntimeConfig(
            qdrant_url="http://example:6333",
            qdrant_collection="mycoll",
            retrieval_mode="dense_only",
        )
    )

    retriever = runtime.build_retriever()

    assert captured == {"collection": "mycoll", "url": "http://example:6333"}
    assert len(retriever.candidate_retrievers) == 1
    assert isinstance(retriever.candidate_retrievers[0], SemanticCandidateRetriever)
