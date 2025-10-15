"""
A unified indexer that uses a VectorDB instance to store embeddings.
"""
from typing import Any, Dict, Iterable, List, Optional

from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.vectordb.base import AbstractVectorDB
from episcope.utils.chunker import chunk_paper
from episcope.utils.data_blueprints import StructuredSection, PaperMetadata
import numpy as np

class Indexer:
    """A unified indexer that chunks, embeds, and stores documents in a VectorDB."""

    def __init__(self, db: AbstractVectorDB, embed_model: str = None, batch_size: int = 8):
        self.db = db
        if embed_model != db.get_embedding_model() and db.get_embedding_model() is not None:
            assert False, f"Warning: embed_model '{embed_model}' does not match VectorDB model '{db.get_embedding_model()}'"
        self.embed_model = embed_model
        assert embed_model is not None, "An embedding model 'embed_model' must be specified if this is the first time the VectorDB is used."
        self.embedder = SimplifiedEmbedder(embed_model=embed_model, batch_size=batch_size)

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
        
        self.db.upsert(points, namespace=namespace, embed_model=self.embed_model)

    def index_paper(
        self,
        sections: List[StructuredSection],
        metadata: PaperMetadata,
        paper_id: str,
        min_chunk_size: int = 50,
    ) -> None:
        """Chunk and index a single structured paper."""
        chunks = chunk_paper(sections, metadata, paper_id, min_chunk_size)
        self.index_documents(chunks, namespace=paper_id)
