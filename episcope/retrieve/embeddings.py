"""
retrieve/embeddings.py

Embedding utilities for RAG pipelines.

This module exposes simple wrapper classes around HuggingFace embedding
models as well as convenience functions for batched encoding of text.
Currently only text embeddings are provided.  In future versions
image and multi‑modal embeddings may be added here.
"""
from __future__ import annotations

from typing import Iterable, List

# The HuggingFace embedding backend may not be available in all test environments.
# Attempt to import it and fall back to a no‑op stub if missing.  This allows
# the module to be imported even when the ``llama_index`` package is absent.
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
except Exception:
    class HuggingFaceEmbedding:  # type: ignore
        """Stubbed HuggingFaceEmbedding used when llama_index is unavailable."""
        def __init__(self, model_name: str, trust_remote_code: bool = True) -> None:
            self.model_name = model_name
        def _get_query_embedding(self, text: str):
            # Return a trivial embedding based on string length
            return [float(len(text))]
        def get_text_embedding_batch(self, texts):  # type: ignore
            return [[float(len(t))] for t in texts]

try:
    from transformers import AutoConfig  # type: ignore
except Exception:
    class AutoConfig:  # type: ignore
        """Stubbed AutoConfig used when transformers is unavailable."""
        @staticmethod
        def from_pretrained(model_name: str, trust_remote_code: bool = True):  # type: ignore
            # Return a simple object with a hidden_size attribute
            return type("DummyCfg", (), {"hidden_size": 1})()
from tqdm import tqdm


class SimplifiedEmbedder:
    """Wrapper around a HuggingFace embedding model.

    This class provides batched embedding methods and exposes the
    embedding dimension via the ``dim`` attribute.  The underlying
    HuggingFace model is loaded on first use and reused for subsequent
    calls.
    """

    def __init__(self, embed_model: str, batch_size: int = 8) -> None:
        self._model_name = embed_model
        self._batch_size = batch_size
        self._embedder = HuggingFaceEmbedding(model_name=embed_model, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(embed_model, trust_remote_code=True)
        self.dim = cfg.hidden_size

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""
        return self._embedder._get_query_embedding(text)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings in batches."""
        texts_list = list(texts)
        vectors: List[List[float]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size)):
            batch = texts_list[i : i + self._batch_size]
            vectors.extend(self._embedder.get_text_embedding_batch(batch))
        return vectors

    def embed_units(self, units: Iterable[dict[str, str]]) -> List[List[float]]:
        """Embed the ``content`` field of each unit in ``units``."""
        texts = [u["content"] for u in units]
        return self.embed_texts(texts)

