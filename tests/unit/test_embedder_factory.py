from __future__ import annotations

import sys
import types

import pytest

from episcope.rag.embeddings import factory
from episcope.rag.embeddings.factory import EmbedderFactory


class _FakeEmbedder:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _patch_providers(monkeypatch):
    # Swap real embedder classes for a cheap fake so dispatch tests never
    # construct a real model or hit the network.
    for name in list(factory.EMBEDDER_PROVIDERS):
        monkeypatch.setitem(factory.EMBEDDER_PROVIDERS, name, _FakeEmbedder)


def test_detect_provider_common_cases() -> None:
    detect = EmbedderFactory._detect_provider
    # Covered by the lightweight fastembed backend: no torch required.
    assert detect("sentence-transformers/all-MiniLM-L6-v2") == "fastembed"
    # Not in fastembed's curated list: falls back to the huggingface backend.
    assert detect("intfloat/e5-small-v2") == "huggingface"
    assert detect("gemini-embedding-001") == "gemini"
    assert detect("models/embedding-001") == "gemini"
    assert detect("text-embedding-3-small") == "openai"
    assert detect("text-embedding-ada-002") == "openai"


def test_detect_provider_raises_on_ambiguous_or_unknown() -> None:
    # These used to silently become Ollama models.
    with pytest.raises(ValueError):
        EmbedderFactory._detect_provider("nomic-embed-text")
    with pytest.raises(ValueError):
        EmbedderFactory._detect_provider("text-embedding-004")  # Gemini vs OpenAI


def test_explicit_provider_dispatch() -> None:
    embedder = EmbedderFactory.get_embedder("nomic-embed-text", provider="ollama")
    assert isinstance(embedder, _FakeEmbedder)
    assert embedder.model == "nomic-embed-text"


def test_auto_detect_then_dispatch() -> None:
    embedder = EmbedderFactory.get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(embedder, _FakeEmbedder)


def test_env_var_selects_provider(monkeypatch) -> None:
    monkeypatch.setenv("EPISCOPE_EMBED_PROVIDER", "ollama")
    embedder = EmbedderFactory.get_embedder("anything-unrecognized")
    assert isinstance(embedder, _FakeEmbedder)


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbedderFactory.get_embedder("whatever", provider="bogus")


def test_auto_without_match_raises(monkeypatch) -> None:
    monkeypatch.delenv("EPISCOPE_EMBED_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Could not infer"):
        EmbedderFactory.get_embedder("nomic-embed-text")


def test_huggingface_dispatch_is_lazy_and_works_when_available(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(
        HuggingFaceEmbedder=_FakeEmbedder,
        HuggingFaceSparseEmbedder=_FakeEmbedder,
        HuggingFaceLateEmbedder=_FakeEmbedder,
    )
    monkeypatch.setitem(
        sys.modules, "episcope.rag.embeddings.huggingface", fake_module
    )
    embedder = EmbedderFactory.get_embedder(
        "intfloat/e5-small-v2", provider="huggingface"
    )
    assert isinstance(embedder, _FakeEmbedder)
    assert embedder.model == "intfloat/e5-small-v2"


def test_huggingface_dispatch_missing_torch_raises_local_ml_hint(monkeypatch) -> None:
    monkeypatch.delitem(
        sys.modules, "episcope.rag.embeddings.huggingface", raising=False
    )
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "" and fromlist and "huggingface" in fromlist and level == 1:
            raise ImportError("simulated missing torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="epi-scope\\[local-ml\\]"):
        EmbedderFactory.get_embedder("intfloat/e5-small-v2", provider="huggingface")
