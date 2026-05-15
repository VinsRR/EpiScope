from abc import ABC, abstractmethod
from typing import Any

from episcope.rag.generation.base import Generator
from episcope.rag.retrieval.base import BaseRetriever


class AbstractRAG(ABC):
    """Abstract base class for a RAG workflow."""

    def __init__(self, retriever: BaseRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    @abstractmethod
    def run(self, paper_id: str, metadata: Any = None) -> Any:
        """Run the RAG workflow."""
        pass
