"""
retrieve/rag/text.py

Implementation of a text‑only retrieval‑augmented generation pipeline.

This class mirrors the original ``TextRAG`` from the ``rag_tool`` project
but has been refactored to improve separation of concerns and
configurability.  It uses the ``SimplifiedEmbedder`` to compute dense
embeddings, a Qdrant vector store via ``UnifiedQdrantIndex`` and
supports optional HYDE query augmentation.
"""
from __future__ import annotations

import os
import uuid
from typing import Iterable, List, Optional, Sequence

from qdrant_client.http import models

from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.rag.retrieval.components.hyde import HYDE
from episcope.rag.retrieval.utils import validate_token_budget, find_pathogen_keyword
from episcope.rag.indexing.unified_db import UnifiedQdrantIndex
from episcope.rag.main import AbstractRAG
from episcope.text_only import (
    EMBED_MODEL as DEFAULT_EMBED_MODEL,
    SPARSE_EMBED_MODEL as DEFAULT_SPARSE_EMBED_MODEL,
    LATE_INTERACTION_MODEL as DEFAULT_LATE_INTERACTION_MODEL,
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    QDRANT_URL as DEFAULT_Q_URL,
    QDRANT_TIMEOUT as DEFAULT_Q_TIMEOUT,
    COLLECTION_NAME as DEFAULT_COLLECTION,
    DISTANCE as DEFAULT_DISTANCE,
    USER_PROMPT as DEFAULT_USER_PROMPT,
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    GENERATOR_MODEL as DEFAULT_GEN_MODEL,
    BINARY_QUANTIZATION as DEFAULT_BINARY_QUANTIZATION,
)

# Optional imports for sparse and late interaction embeddings
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

import ollama
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextRAG(AbstractRAG):
    """Retrieval‑augmented generation over plain text documents."""

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        sparse_embed_model: str = DEFAULT_SPARSE_EMBED_MODEL,
        late_interaction_model: str = DEFAULT_LATE_INTERACTION_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        qdrant_url: str = DEFAULT_Q_URL,
        qdrant_api_key: Optional[str] = None,
        qdrant_timeout: int = DEFAULT_Q_TIMEOUT,
        collection_name: str = DEFAULT_COLLECTION,
        distance: str = DEFAULT_DISTANCE,
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        gen_model: str = DEFAULT_GEN_MODEL,
        hyde_model_name: Optional[str] = "tinyllama:1.1b",
        binary_quantization: bool = DEFAULT_BINARY_QUANTIZATION,
    ) -> None:
        self.embedder = SimplifiedEmbedder(embed_model=embed_model, batch_size=batch_size)
        self.sparse_embedder = SparseTextEmbedding(sparse_embed_model)
        self.late_embedder = LateInteractionTextEmbedding(late_interaction_model)
        # Approximate dimension for late interaction (could be inferred)
        self.dim_late_interaction = 128

        self.db = UnifiedQdrantIndex(
            collection=collection_name,
            dim=self.embedder.dim,
            dim_late_interaction=self.dim_late_interaction,
            url=qdrant_url,
            api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
            timeout=qdrant_timeout,
            distance=distance,
        )

        # Create collection with multivector and sparse support
        self.db.create_collection(
            multivector=False,
            sparse=True,
            late_interaction=True,
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
            if binary_quantization
            else None,
        )

        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.gen_model = gen_model
        self.hyde = HYDE(hyde_model_name) if hyde_model_name else None
        self._namespace_uuid = uuid.UUID("f9a7c0b0-b8b3-4c3d-8a5a-9f5c7b1e2d0a")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _extract_units_from_pdf(self, filepath: str) -> List[dict[str, str]]:
        """Extract text units from a PDF file using unstructured.partition.pdf.

        Tables are converted to plain text; images are ignored.  Each unit
        is returned as a dict with fields ``content``, ``file``, ``type`` and
        ``id``.
        """
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Table, CompositeElement
        from tqdm import tqdm

        chunks = partition_pdf(
            filepath,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=[],
            extract_image_block_to_payload=False,
            chunking_strategy="by_title",
            max_characters=3000,
            combine_text_under_n_chars=650,
            new_after_n_chars=2000,
        )

        units: List[dict[str, str]] = []
        for chunk in tqdm(chunks, desc="Extracting text units..."):
            text_content = None
            if isinstance(chunk, Table) or isinstance(chunk, CompositeElement):
                text_content = chunk.text
            if text_content and text_content.strip():
                units.append(
                    {
                        "content": text_content.strip(),
                        "file": filepath,
                        "type": "text",
                        "id": uuid.uuid5(self._namespace_uuid, text_content.strip()),
                        "pathogen": find_pathogen_keyword(filepath) or "",
                    }
                )
        logger.info(f"Extracted {len(units)} text units from {filepath}")
        return units

    def index(self, source: str | Iterable[str]) -> None:
        """Index a PDF file or directory of PDFs."""
        from os import path, walk
        if isinstance(source, str):
            if source.lower().endswith(".pdf"):
                self._index_file(source)
            elif path.isdir(source):
                self._index_directory(source)
            else:
                raise ValueError("Unsupported source type; provide a PDF file or directory")
        else:
            # Assume iterable of PDF paths
            for item in source:
                self.index(item)

    def _index_file(self, filename: str) -> None:
        logger.info(f"Indexing file: {filename}")
        units = self._extract_units_from_pdf(filename)
        dense_embs = self.embedder.embed_units(units)
        docs = [u["content"] for u in units]
        sparse_embs = list(self.sparse_embedder.embed(doc for doc in docs))
        late_embs = list(self.late_embedder.embed(doc for doc in docs))
        points = []
        for unit, d_emb, s_emb, l_emb in zip(units, dense_embs, sparse_embs, late_embs):
            points.append(
                models.PointStruct(
                    id=str(unit["id"]),
                    vector={
                        "dense": d_emb,
                        "sparse": s_emb.as_object(),
                        "late_interaction": l_emb,
                    },
                    payload={k: v for k, v in unit.items() if k != "id"},
                )
            )
        self.db.upsert(points)
        logger.info(f"Successfully indexed {len(points)} units from {filename}")

    def _index_directory(self, directory: str) -> None:
        from os import walk
        for dp, _, files in walk(directory):
            for f in files:
                if f.lower().endswith(".pdf"):
                    self._index_file(os.path.join(dp, f))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10, pathogen: Optional[str] = None, use_hyde: bool = False) -> List[dict[str, str]]:
        if use_hyde and self.hyde:
            query = self.hyde.generate(query)
        # Build query embeddings
        dense_q = self.embedder.embed_text(query)
        sparse_q = next(self.sparse_embedder.query_embed(query))
        prefetch_filter = None
        if pathogen:
            prefetch_filter = models.Filter(
                must=[models.FieldCondition(key="pathogen", match=models.MatchValue(value=pathogen))]
            )
        prefetch = [
            models.Prefetch(query=dense_q, using="dense", limit=top_k * 3, filter=prefetch_filter),
            models.Prefetch(query=models.SparseVector(**sparse_q.as_object()), using="sparse", limit=top_k * 3, filter=prefetch_filter),
        ]
        results = self.db.client.query_points(
            collection_name=self.db.collection,
            prefetch=prefetch,
            query=next(self.late_embedder.query_embed(query)),
            using="late_interaction",
            with_payload=True,
            limit=top_k,
        )
        dict_results = results.model_dump()["points"]
        sorted_results = sorted(dict_results, key=lambda x: x["score"], reverse=True)
        final_ordered_payloads: List[Optional[dict[str, str]]] = [None] * top_k
        left_idx = 0
        right_idx = top_k - 1
        for i in range(top_k):
            if i % 2 == 0:
                final_ordered_payloads[left_idx] = sorted_results[i]["payload"]
                left_idx += 1
            else:
                final_ordered_payloads[right_idx] = sorted_results[i]["payload"]
                right_idx -= 1
        return [p for p in final_ordered_payloads if p is not None]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, query: str, contexts: Sequence[dict[str, str]]) -> str:
        snippet_list = "\n".join(f"[{i+1}] {item['content']}" for i, item in enumerate(contexts))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.format(
                    question=query,
                    num_snippets=len(contexts),
                    snippet_list=snippet_list,
                ),
            },
        ]
        validate_token_budget(self.gen_model, messages)
        resp = ollama.chat(model=self.gen_model, messages=messages)
        return resp.message.content.strip()

