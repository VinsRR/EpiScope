from typing import Iterable, List
from .base import Embedder
from episcope.clients import GeminiClient


class GeminiEmbedder(Embedder):
    def __init__(
        self,
        model: str = "gemini-embedding-001",  # "embedding-001",# "models/embedding-001",
        client: GeminiClient = None,
        # 3072 is the maximum available dimension for Gemini embeddings
        # Nevertheles, they use what they call "Matrioshka" embeddings
        # where the the model is trained to store the most important semantic information in the "earlier" parts of the vector,
        # a 768-dimensional slice of a 3072-vector is nearly as accurate as the full vector.
        fixed_dim: int = None,
    ):
        self._model = model
        self._client = client or GeminiClient()
        assert fixed_dim is None or (0 < fixed_dim <= 3072), (
            "fixed_dim must be between 1 and 3072"
        )
        self._dim = fixed_dim
        if self._dim is None:
            self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_texts_with_task(
        self, texts: Iterable[str], *, task_type: str
    ) -> List[List[float]]:
        return self._client.embed(
            list(texts),
            model=self._model,
            output_dimensionality=self._dim,
            task_type=task_type,
        )

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts_with_task(
            [text], task_type="RETRIEVAL_QUERY"
        )[0]

    def embed_queries(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed_texts_with_task(texts, task_type="RETRIEVAL_QUERY")

    def embed_document(self, text: str) -> List[float]:
        return self._embed_texts_with_task(
            [text], task_type="RETRIEVAL_DOCUMENT"
        )[0]

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed_texts_with_task(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_document(text)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return self.embed_documents(texts)
