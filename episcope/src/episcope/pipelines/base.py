from abc import ABC, abstractmethod
from typing import Any

from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever


class AbstractRAG(ABC):
    """Abstract base class for a RAG pipeline."""

    def __init__(self, retriever: AbstractRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the RAG pipeline."""
        pass