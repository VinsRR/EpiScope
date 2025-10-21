from __future__ import annotations
from typing import Iterable, List
from tqdm import tqdm
from .base import Embedder

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    class HuggingFaceEmbedding:
        """Stubbed HuggingFaceEmbedding used when llama_index is unavailable."""
        def __init__(self, model: str, trust_remote_code: bool = True) -> None:
            self.model = model
        def _get_query_embedding(self, text: str):
            return [float(len(text))]
        def get_text_embedding_batch(self, texts):
            return [[float(len(t))] for t in texts]

try:
    from transformers import AutoConfig
except ImportError:
    class AutoConfig:
        """Stubbed AutoConfig used when transformers is unavailable."""
        @staticmethod
        def from_pretrained(model: str, trust_remote_code: bool = True):
            return type("DummyCfg", (), {"hidden_size": 1})()

class HuggingFaceEmbedder(Embedder):
    """Wrapper around a HuggingFace embedding model."""

    def __init__(self, model: str, batch_size: int = 8) -> None:
        self._model = model
        self._batch_size = batch_size
        self._embedder = HuggingFaceEmbedding(model_name=model, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # self._dim = cfg.hidden_size
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""
        return self._embedder._get_query_embedding(text)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings in batches."""
        texts_list = list(texts)
        vectors: List[List[float]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding texts"):
            batch = texts_list[i : i + self._batch_size]
            vectors.extend(self._embedder.get_text_embedding_batch(batch))
        return vectors
