"""
A unified indexer that uses a VectorDB instance to store embeddings.
"""
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from episcope.rag.embeddings.base import Embedder
from episcope.rag.indexing.chunking import Chunker, NoChunker, chunk_paper
from episcope.schemas import PaperMetadata, StructuredSection
from episcope.vectordb.base import AbstractVectorDB


class Indexer:
    """A unified indexer that chunks, embeds, and stores documents in a VectorDB."""

    def __init__(self, db: AbstractVectorDB, embedder: Embedder, chunker: Optional[Chunker] = None):
        self.db = db
        self.embedder = embedder
        self.chunker = chunker or NoChunker()

    def index_documents(self, docs: Iterable[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        docs_list = list(docs)
        if not docs_list:
            return

        texts = [d.get("content", "") for d in docs_list]
        embeddings = self.embedder.embed_texts(texts)
        
        embeddings = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        points = []
        for vec, doc in zip(embeddings, docs_list):
            payload = {k: v for k, v in doc.items()}
            points.append({"vector": vec.tolist(), "payload": payload, "id": doc.get("id")})
        
        self.db.upsert(
            points,
            namespace=namespace,
            embed_model=self.embedder.model_name,
            chunking_config=self.chunker.config
        )

    def index_paper(
        self,
        sections: List[StructuredSection],
        metadata: PaperMetadata,
        paper_id: str,
    ) -> None:
        """Chunk and index a single structured paper."""
        chunks = chunk_paper(self.chunker, sections, metadata, paper_id)
        self.index_documents(chunks, namespace=paper_id)