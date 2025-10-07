"""Structured per‑paper indexer implementation.

This module defines a simple per‑paper indexer that builds an
in‑memory index over the structured sections of a paper.  It is
designed for use by the PrecisionMiner pipeline where each paper
requires its own dedicated index rather than a global project
index.  Embeddings are computed via the same ``SimplifiedEmbedder``
used by the global FAISS indexer, but if transformer models are
unavailable a trivial length‑based embedding is used.  Chunking
utilities from the legacy :class:`EmbeddingIndexer` are preserved
here to avoid code duplication and satisfy existing tests.

The index maintains a mapping of ``paper_id`` to both the
embedding matrix and associated metadata for each chunk.  When
searching, only the index for the specified paper is consulted.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.interfaces import AbstractIndexer
from ..retrieve.embeddings import SimplifiedEmbedder
from ..parse.blueprints.data_blueprints import StructuredSection, PaperMetadata

logger = logging.getLogger(__name__)


class PaperIndexer(AbstractIndexer):
    """An indexer for building and querying per‑paper indices.

    This class exposes a similar API to :class:`FaissIndexer` but
    scopes all documents to a single ``paper_id``.  It performs
    lightweight paragraph‑based chunking on structured sections and
    uses the configured embedder to compute vector representations.
    If the embedder cannot be instantiated (due to missing models),
    a deterministic length‑based embedding is used instead.  Each
    chunk stores its originating section information for later
    provenance.
    """

    def __init__(self, embed_model: str = "jinaai/jina-embeddings-v3" , batch_size: int = 8, *, min_chunk_size: int = 50) -> None: # "distilbert-base-uncased"
        self.embedder: SimplifiedEmbedder
        try:
            self.embedder = SimplifiedEmbedder(embed_model=embed_model, batch_size=batch_size)
        except Exception:
            # Fall back to a dummy embedder that encodes text length
            self.embedder = None  # type: ignore
        self.min_chunk_size = max(min_chunk_size, 1)
        # Map of paper_id -> (embeddings matrix, list of metadata dicts)
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # AbstractIndexer implementation
    # ------------------------------------------------------------------

    def index_documents(self, docs: Iterable[Dict[str, Any]], *, namespace: Optional[str] = None) -> None:
        """Index a collection of arbitrary document dictionaries.

        This method allows the ``PaperIndexer`` to satisfy the
        ``AbstractIndexer`` interface.  Each document must contain
        ``paper_id`` and ``content`` keys.  For each document, a
        single embedding is computed.  Multiple documents for the
        same paper will accumulate embeddings and metadata.
        """
        docs_list = list(docs)
        for doc in docs_list:
            paper_id = doc.get("paper_id")
            text = doc.get("content", "")
            if not paper_id or not text:
                continue
            emb = self._embed_text(text)
            self._embeddings.setdefault(paper_id, np.empty((0, len(emb)), dtype="float32"))
            self._embeddings[paper_id] = np.vstack([self._embeddings[paper_id], emb[np.newaxis, :]])
            self._metadata.setdefault(paper_id, []).append(doc)

    def search(self, query: str, *, top_k: int = 5, namespace: Optional[str] = None) -> Sequence[Dict[str, Any]]:
        """Search a paper‑specific index with the given query.

        The ``namespace`` argument is ignored for this indexer; the
        caller must specify the ``paper_id`` inside the query
        metadata when using ``index_paper``.  If multiple papers
        have been indexed via :meth:`index_documents`, the search
        results are aggregated across all papers and sorted by
        similarity.  Returns at most ``top_k`` results with
        associated metadata and a ``score`` field.
        """
        q_emb = self._embed_text(query)
        results: List[Dict[str, Any]] = []
        for paper_id, embs in self._embeddings.items():
            if embs.size == 0:
                continue
            sims = (embs @ q_emb).astype(float)
            # Determine top matches for this paper
            top_idx = sims.argsort()[-top_k:][::-1]
            for idx in top_idx:
                meta = self._metadata[paper_id][idx]
                # Include the paper_id and score; normalise 'text' to 'content' for consistency
                result = {**meta, "score": float(sims[idx]), "paper_id": paper_id}
                # Normalise text field to content if needed
                if "text" in result and "content" not in result:
                    result["content"] = result["text"]
                results.append(result)
        # Sort overall results by score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # High‑level API for structured papers
    # ------------------------------------------------------------------

    def index_paper(self, sections: List[StructuredSection], metadata: PaperMetadata, paper_id: str) -> None:
        """Index a single paper's sections and metadata.

        This convenience method performs paragraph‑based chunking on
        each structured section, prepends the abstract as an extra
        chunk if provided, computes embeddings and stores
        per‑paper data for retrieval.  Any existing index for
        ``paper_id`` will be overwritten.
        """
        chunks: List[Dict[str, Any]] = []
        # Add metadata abstract as first chunk
        if metadata and metadata.abstract:
            chunks.append({
                "text": metadata.abstract,
                "section_title": "Abstract",
                "section_type": "Abstract",
                "paper_id": paper_id,
                "is_metadata": True,
            })
        # Process sections
        for section in sections:
            section_chunks = self._paragraph_chunking(section)
            for chunk in section_chunks:
                chunk_meta = {
                    "text": chunk["text"],
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "paper_id": paper_id,
                    "is_metadata": False,
                }
                chunks.append(chunk_meta)
        # Compute embeddings and store
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = np.array([self._embed_text(t) for t in texts], dtype="float32")
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms
        self._embeddings[paper_id] = embeddings
        self._metadata[paper_id] = chunks
        logger.debug(f"Indexed {len(chunks)} chunks for paper {paper_id}")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single piece of text into a vector.

        If the configured embedder is unavailable, a deterministic
        one‑dimensional embedding based on token count is used.  The
        return value is always a one‑dimensional numpy array of
        length ``d``, where ``d`` is determined by the underlying
        embedder.  For the dummy embedder ``d`` is 1.
        """
        if self.embedder is None:
            # Return a 1D vector containing the number of tokens
            return np.array([len(text.split())], dtype="float32")
        try:
            return np.array(self.embedder.embed_text(text), dtype="float32")
        except Exception:
            # Fallback on error
            return np.array([len(text.split())], dtype="float32")

    # ------------------------------------------------------------------
    # Legacy chunking helpers (mirroring parse.indexing.embeddings)
    # ------------------------------------------------------------------
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation heuristics.

        This method mirrors the behaviour of the legacy
        ``EmbeddingIndexer`` to satisfy existing tests.  It simply
        splits on `.`, `!` and `?` followed by whitespace while
        preserving the delimiter.
        """
        # Use regex to split while keeping punctuation with the sentence
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        parts = sentence_endings.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _paragraph_chunking(self, section: StructuredSection) -> List[Dict[str, Any]]:
        """Produce paragraph chunks for a structured section.

        Chunks are created for paragraphs that exceed ``min_chunk_size``
        characters and contain more than five words.  Each returned
        dict contains only a ``text`` key; additional metadata is
        added in :meth:`index_paper`.
        """
        chunks: List[Dict[str, Any]] = []
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > self.min_chunk_size and len(paragraph.split()) > 5:
                chunks.append({"text": paragraph})
        return chunks
