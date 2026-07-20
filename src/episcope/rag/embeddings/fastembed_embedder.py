from __future__ import annotations

from typing import Iterable, List, Optional

from .base import Embedder

# Curated subset of fastembed.TextEmbedding.list_supported_models() that
# overlaps with models EpiScope has historically shipped as defaults. This is
# intentionally conservative rather than the full fastembed catalog: it's
# the set we've verified produces embeddings matching the huggingface
# backend's output for the same model name.
FASTEMBED_SUPPORTED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
}


class FastEmbedEmbedder(Embedder):
    """Dense text embedder backed by fastembed (ONNX runtime, no torch)."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        if model not in FASTEMBED_SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model}' is not in the list of models supported by the "
                f"lightweight fastembed backend: {sorted(FASTEMBED_SUPPORTED_MODELS)}. "
                f"Pass provider='huggingface' (needs `pip install epi-scope[local-ml]`) "
                f"to use other models."
            )
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the default lightweight embedding "
                "backend but isn't installed. Install it with `pip install "
                "fastembed`, or `pip install epi-scope[local-ml]` for the "
                "torch-based backend instead."
            ) from exc

        self._model = model
        self._batch_size = batch_size
        self._normalize = normalize
        self._encoder = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        vectors = list(self._encoder.embed(texts_list, batch_size=self._batch_size))
        if self._normalize:
            import numpy as np

            vectors = [v / (np.linalg.norm(v) or 1.0) for v in vectors]
        return [v.tolist() for v in vectors]
