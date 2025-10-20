from abc import ABC, abstractmethod
from typing import Iterable, List

class Embedder(ABC):
    """Abstract base class for embedders."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the embeddings."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""
        pass

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings."""
        pass

    def embed_units(self, units: Iterable[dict[str, str]]) -> List[List[float]]:
        """Embed the ``content`` field of each unit in ``units``."""
        texts = [u["content"] for u in units]
        return self.embed_texts(texts)
