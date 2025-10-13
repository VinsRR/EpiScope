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

from episcope.rag.interfaces import AbstractIndexer, AbstractRetriever
from episcope.rag.retrieval.hyde import HYDE


class TextRetriever(AbstractRetriever):
    """Standard semantic retriever for Explorer mode.

    Instances of this class combine a document indexer with an
    optional HYDE model.  When a HYDE instance is supplied, the
    retriever will first generate hypothetical documents from the
    query and append them to the query to improve recall.  The
    final query is then passed to the indexer to retrieve the top
    matching contexts.
    """

    def __init__(self, indexer: AbstractIndexer, hyde: Optional[HYDE] = None) -> None:
        self.indexer = indexer
        self.hyde = hyde

    def retrieve(self, query: str, *, top_k: int = 5, namespace: Optional[str] = None, **kwargs: Any) -> Sequence[Dict[str, Any]]:
        # If HYDE is provided, augment the query
        final_query = query
        if self.hyde is not None:
            # generate a single hypothetical document and append
            hyp = self.hyde.generate(query)
            # Append separated by newline for clarity
            if hyp:
                final_query = f"{query}\n\n{hyp}"
        # Delegate to the indexer
        results = self.indexer.search(final_query, top_k=top_k, namespace=namespace)
        # Ensure each result contains a 'content' key expected by generators
        contexts: List[Dict[str, Any]] = []
        for res in results:
            # Copy metadata and rename 'text' or 'content' to 'content'
            content = res.get("text") or res.get("content") or ""
            ctx = {**res, "content": content}
            contexts.append(ctx)
        return contexts
