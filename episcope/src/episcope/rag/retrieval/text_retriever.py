"""Text retriever implementation for the Explorer mode.

This module defines a simple retriever that delegates semantic search
to an underlying :class:`AbstractIndexer`.  It optionally performs
query enhancement via the HYDE technique when a HYDE instance is
provided.  The retriever returns a list of context dictionaries
containing the text content and any associated metadata stored in
the indexer.  Scores are preserved to allow downstream selection
logic.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, List, Optional

from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.retrieval.components.hyde import HYDE
from episcope.rag.vectordb.base import AbstractVectorDB
from episcope.rag.embeddings import SimplifiedEmbedder

class TextRetriever(AbstractRetriever):
    """Standard semantic retriever for Explorer mode."""

    def __init__(self, db: AbstractVectorDB, embed_model: str = "distilbert-base-uncased", hyde: Optional[HYDE] = None) -> None:
        self.db = db
        self.embedder = SimplifiedEmbedder(embed_model=embed_model)
        self.hyde = hyde

    def retrieve(self, query: str, *, top_k: int = 5, namespace: Optional[str] = None, **kwargs: Any) -> Sequence[Dict[str, Any]]:
        final_query = query
        if self.hyde is not None:
            hyp = self.hyde.generate(query)
            if hyp:
                final_query = f"{query}\\n\\n{hyp}"
        
        query_vector = self.embedder.embed_text(final_query)
        
        results = self.db.search(query_vector, top_k=top_k, namespace=namespace)
        
        contexts: List[Dict[str, Any]] = []
        for res in results:
            content = res.get("text") or res.get("content") or ""
            ctx = {**res, "content": content}
            contexts.append(ctx)
        return contexts
