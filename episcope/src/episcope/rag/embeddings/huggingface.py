from __future__ import annotations
from typing import Iterable, List
from tqdm import tqdm

from .base import Embedder

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoConfig

class HuggingFaceEmbedder(Embedder):
    """Wrapper around a HuggingFace embedding model."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 8) -> None:
        self._model = model
        self._batch_size = batch_size
        # pass model_name to the underlying class
        self._embedder = HuggingFaceEmbedding(model_name=model)
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # if you want a true hidden size: uncomment next line
        # self._dim = cfg.hidden_size
        # but as fallback, we embed a dummy text
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""
        return self._embedder.get_text_embedding(text)  # or _get_query_embedding if your version uses that

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings in batches."""
        texts_list = list(texts)
        vectors: List[List[float]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding texts"):
            batch = texts_list[i : i + self._batch_size]
            vectors.extend(self._embedder.get_text_embedding_batch(batch))
        return vectors
