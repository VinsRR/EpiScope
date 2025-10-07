"""Specialized FAISS indexer for the parsing pipeline.

This module provides the `EmbeddingIndexer`, a subclass of
`episcope.index.faiss_indexer.FaissIndexer`. It is tailored to the
needs of the processing pipeline, providing a `create_index` method
that handles structured paper data, performs chunking, and persists
the resulting index and chunk files to disk.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from ...index.faiss_indexer import FaissIndexer
from ..blueprints.data_blueprints import StructuredSection, PaperMetadata
from .chunking import paragraph_chunking

logger = logging.getLogger(__name__)


class EmbeddingIndexer(FaissIndexer):
    """
    A specialized FaissIndexer that understands structured paper data
    and provides a pipeline-compatible `create_index` method.
    """

    def __init__(self,
                 model_name: str = "jinaai/jina-embeddings-v3",
                 min_chunk_size: int = 50,
                 **kwargs: Any) -> None:
        super().__init__(embed_model=model_name, batch_size=kwargs.get("batch_size", 8))
        self.min_chunk_size = min_chunk_size
        logger.info("Initialized specialized EmbeddingIndexer for the pipeline.")

    def create_index(
        self,
        sections: List[StructuredSection],
        metadata: PaperMetadata,
        output_dir: str,
        paper_id: str
    ) -> str | None:
        """
        Chunks a structured paper, creates a FAISS index, and saves both to disk.
        """
        chunks = self._create_chunks(sections, metadata, paper_id)
        if not chunks:
            logger.warning(f"No chunks were created for paper {paper_id}, index not generated.")
            return None

        # Use the parent FaissIndexer to create the in-memory index
        self.index_documents(chunks, namespace=paper_id)

        # Use the parent FaissIndexer to save the index and chunks to files
        return self.save_index_files(namespace=paper_id, output_dir=output_dir, paper_id=paper_id)

    def _create_chunks(
        self,
        sections: List[StructuredSection],
        metadata: PaperMetadata,
        paper_id: str
    ) -> List[Dict[str, Any]]:
        """Creates a list of chunks from paper sections and metadata."""
        chunks: List[Dict[str, Any]] = []
        if metadata and metadata.abstract:
            chunks.append({
                "text": metadata.abstract,
                "content": metadata.abstract,
                "section_title": "Abstract",
                "section_type": "Abstract",
                "paper_id": paper_id,
                "is_metadata": True,
            })

        for section in sections:
            section_chunks = paragraph_chunking(section, self.min_chunk_size)
            for chunk in section_chunks:
                chunk_meta = {
                    "text": chunk["text"],
                    "content": chunk["text"],
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "paper_id": paper_id,
                    "is_metadata": False,
                }
                chunks.append(chunk_meta)
        return chunks