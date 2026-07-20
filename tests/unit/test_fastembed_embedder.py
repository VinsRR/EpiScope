from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from episcope.rag.embeddings.fastembed_embedder import (
    FASTEMBED_SUPPORTED_MODELS,
    FastEmbedEmbedder,
)


class _FakeTextEmbedding:
    def __init__(self, model_name: str, cache_dir=None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts, batch_size=32):
        for text in texts:
            yield np.array([float(len(text)), 1.0, 0.0])


@pytest.fixture(autouse=True)
def _patch_fastembed(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding)
    )


def test_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="not in the list"):
        FastEmbedEmbedder(model="not-a-real-model")


def test_default_model_is_supported() -> None:
    assert "sentence-transformers/all-MiniLM-L6-v2" in FASTEMBED_SUPPORTED_MODELS
    embedder = FastEmbedEmbedder()
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.dim == 3


def test_embed_texts_normalizes_vectors() -> None:
    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
    vectors = embedder.embed_texts(["hello", "a longer piece of text"])
    assert len(vectors) == 2
    for vector in vectors:
        norm = sum(v * v for v in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_embed_texts_empty_list_returns_empty() -> None:
    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
    assert embedder.embed_texts([]) == []


def test_embed_text_matches_embed_texts_single_item() -> None:
    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5")
    single = embedder.embed_text("hello")
    batch = embedder.embed_texts(["hello"])[0]
    assert single == batch
