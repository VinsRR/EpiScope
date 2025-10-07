"""
retrieve/rag/factory.py

Factory for instantiating retrieval‑augmented generation (RAG) methods.

Concrete RAG implementations live in ``retrieve/rag/`` and should be
registered here to make them discoverable via configuration.  The
``get`` method returns an instance of the requested RAG class.
"""
from __future__ import annotations

from typing import Any, Type

from .text import TextRAG
# TODO: add ColPaliRAG and GraphRAG implementations when ported


class RAGFactory:
    """Factory for RAG methods."""

    _registry: dict[str, Type] = {
        "text": TextRAG,
        # "colpali": ColPaliRAG,  # to be implemented
        # "graph": GraphRAG,      # to be implemented
    }

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown RAG method '{name}'")
        return cls._registry[key](**kwargs)

