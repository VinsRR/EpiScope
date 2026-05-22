from typing import Iterable, List, Optional
from .base import Embedder
from episcope.clients import OpenAIClient


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, client: Optional[OpenAIClient] = None):
        self._model = model
        self._client = client or OpenAIClient()
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        return self._client.embed([text], model=self._model)[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return self._client.embed(list(texts), model=self._model)
