from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from episcope.rag.embeddings.base import Embedder
from episcope.rag.indexing.chunking import Chunker, NoChunker, chunk_paper
from episcope.schemas import PaperMetadata, StructuredSection
from episcope.vectordb.base import AbstractVectorDB


class Indexer:
    """Chunks, embeds, and stores documents with optional dense/sparse/late representations."""

    def __init__(
        self,
        db: AbstractVectorDB,
        embedder: Optional[Embedder] = None,
        sparse_embedder: Optional[Embedder] = None,
        late_embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
    ):
        self.db = db
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.late_embedder = late_embedder
        self.chunker = chunker or NoChunker()

    def _validate_embedders(self) -> None:
        caps = self.db.capabilities()
        if caps.get("dense") and self.embedder is None:
            raise ValueError(
                "DB requires dense vectors but no dense embedder was provided."
            )
        if caps.get("sparse") and self.sparse_embedder is None:
            raise ValueError(
                "DB requires sparse vectors but no sparse embedder was provided."
            )
        if caps.get("late") and self.late_embedder is None:
            raise ValueError(
                "DB requires late-interaction vectors but no late embedder was provided."
            )

    def index_documents(
        self, docs: Iterable[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        self._validate_embedders()

        docs_list = list(docs)
        if not docs_list:
            return

        texts = [d.get("content", "") for d in docs_list]
        caps = self.db.capabilities()

        dense_embeddings = None
        sparse_embeddings = None
        late_embeddings = None

        if caps.get("dense"):
            dense_embeddings = self.embedder.embed_texts(texts)
            dense_embeddings = np.array(dense_embeddings, dtype="float32")

            dense_distance = getattr(self.db, "dense_distance", None)
            if (
                dense_distance is not None
                and getattr(dense_distance, "name", str(dense_distance)) == "COSINE"
            ):
                norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True) + 1e-9
                dense_embeddings = dense_embeddings / norms

            dense_embeddings = dense_embeddings.tolist()

        if caps.get("sparse"):
            sparse_embeddings = self.sparse_embedder.embed_texts(texts)

        if caps.get("late"):
            late_embeddings = self.late_embedder.embed_texts(texts)

        embed_models: Dict[str, str] = {}
        if self.embedder is not None:
            embed_models["dense"] = self.embedder.model_name
        if self.sparse_embedder is not None:
            embed_models["sparse"] = self.sparse_embedder.model_name
        if self.late_embedder is not None:
            embed_models["late"] = self.late_embedder.model_name

        points = []
        for i, doc in enumerate(docs_list):
            payload = {k: v for k, v in doc.items()}
            point: Dict[str, Any] = {
                "id": doc.get("id"),
                "payload": payload,
            }

            vectors = {}
            if dense_embeddings is not None:
                vectors["dense"] = dense_embeddings[i]
            if late_embeddings is not None:
                vectors["late"] = late_embeddings[i]
            if vectors:
                point["vectors"] = vectors

            if sparse_embeddings is not None:
                point["sparse_vectors"] = {"sparse": sparse_embeddings[i]}

            points.append(point)

        self.db.upsert(
            points,
            namespace=namespace,
            embed_models=embed_models,
            chunking_config=self.chunker.config,
        )

    def index_paper(
        self,
        sections: List[StructuredSection],
        metadata: PaperMetadata,
        paper_id: str,
    ) -> None:
        chunks = chunk_paper(self.chunker, sections, metadata, paper_id)
        self.index_documents(chunks, namespace=paper_id)
