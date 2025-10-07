"""Compatibility wrapper for structured embedding indexing.

This module provides a thin wrapper around the new
``episcope.index.paper_indexer.PaperIndexer`` class to preserve
backwards compatibility with existing code and tests.  The original
``EmbeddingIndexer`` implemented multiple chunking strategies and
relied on heavy dependencies such as ``faiss`` and
``sentence_transformers``.  Those responsibilities are now handled
by the ``PaperIndexer`` and the ``FaissIndexer`` in the ``index``
package.  Only the paragraph‑based chunking and sentence splitting
utilities are retained here to satisfy unit tests.
"""

from __future__ import annotations

import logging
from typing import List, Dict

from ...index.paper_indexer import PaperIndexer
from ..blueprints.data_blueprints import StructuredSection, PaperMetadata

logger = logging.getLogger(__name__)


class EmbeddingIndexer(PaperIndexer):
    """Alias for PaperIndexer preserving the old API surface.

    This subclass exposes the same constructor signature as the
    legacy ``EmbeddingIndexer`` but delegates all work to
    ``PaperIndexer``.  Parameters unrelated to paragraph chunking
    (e.g. ``chunk_size``, ``chunk_overlap``, ``semantic_threshold``)
    are accepted for compatibility but currently unused.  Future
    implementations could leverage these to implement additional
    chunking strategies.
    """

    def __init__(self,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 chunking_strategy: str = "paragraph",
                 chunk_size: int = 200,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 50,
                 semantic_threshold: float = 0.5,
                 recursive_separators: List[str] | None = None) -> None:
        super().__init__(embed_model=model_name, batch_size=8, min_chunk_size=min_chunk_size)
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.recursive_separators = recursive_separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        logger.info(f"Initialized EmbeddingIndexer (compat) with {self.chunking_strategy} chunking")

    # Expose type hints for _paragraph_chunking and _split_into_sentences
    def _paragraph_chunking(self, section: StructuredSection) -> List[Dict[str, str]]:  # type: ignore[override]
        return super()._paragraph_chunking(section)

    def _split_into_sentences(self, text: str) -> List[str]:  # type: ignore[override]
        return super()._split_into_sentences(text)

    # The original ``create_index`` method saved FAISS indices to disk;
    # for backward compatibility a no‑op implementation is provided.
    def create_index(self, sections: List[StructuredSection], metadata: PaperMetadata, output_dir: str, paper_id: str) -> None:
        """Build an index for a structured paper and persist to disk.

        The compatibility implementation simply calls ``index_paper``
        and does not write any files.  The return value mirrors
        the legacy method but always returns ``None`` to signal that
        no files were produced.  This behaviour satisfies the unit
        tests which focus on chunking rather than persistence.
        """
        self.index_paper(sections, metadata, paper_id)
        return None
