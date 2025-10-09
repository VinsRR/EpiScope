# querying/rag_querier.py
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import faiss
import numpy as np
from ...retrieve.embeddings import SimplifiedEmbedder
from ..blueprints.data_blueprints import Chunk

logger = logging.getLogger(__name__)


class RAGQuerier:
    """Efficient, simple retrieval over per-paper chunks with optional FAISS acceleration.

    - Loads chunks once per instance and caches them.
    - If an index_path is provided, it will try to use FAISS; otherwise it uses in-memory
      semantic ranking (embedding caching) and a small inverted index for keyword scans.
    - Public methods:
        - query_structured_chunks(query, index_path, chunks_path, section_filter, top_k, similarity_threshold)
        - query_keyword_chunks(chunks_path, keywords, top_k)
        - get_section_types(chunks_path)
    """

    def __init__(self, embedder: SimplifiedEmbedder):
        self.embedder = embedder
        # caches keyed by chunks_path
        self._chunks_cache: Dict[str, List[Chunk]] = {}
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._inverted_cache: Dict[str, Dict[str, List[int]]] = {}
        self._faiss_index_cache: Dict[str, faiss.Index] = {}

    # ---------- Loading helpers ----------
    def _load_chunks(self, chunks_path: str) -> List[Chunk]:
        if chunks_path in self._chunks_cache:
            return self._chunks_cache[chunks_path]

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunks file {chunks_path}: {e}")
            return []

        chunks: List[Chunk] = []
        for i, c in enumerate(raw):
            chunks.append(
                Chunk(
                    id=str(c.get("id", i)),
                    section_type=c.get("section_type", c.get("section", "other")),
                    title=c.get("title", ""),
                    text=c.get("text", c.get("content", "")),
                    metadata={k: v for k, v in c.items() if k not in ("text", "content")},
                )
            )
        self._chunks_cache[chunks_path] = chunks
        return chunks

    def _ensure_embeddings(self, chunks_path: str):
        """
        Ensure embeddings are computed for the given chunks path.
        Also normalizes them for cosine similarity and caches them.
        """
        if chunks_path in self._embeddings_cache:
            return
        chunks = self._load_chunks(chunks_path)
        texts = [c.text for c in chunks]
        if not texts:
            self._embeddings_cache[chunks_path] = np.zeros((0, 1))
            return
        # batch encode once and cache
        embs = np.array(self.embedder.embed_texts(texts), dtype="float32")
        # normalize embeddings for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        self._embeddings_cache[chunks_path] = embs

    def _build_inverted(self, chunks_path: str):
        """
        Inverted indexes are built and cached per chunks_path.
        An inverted index maps tokens to chunk indices for quick keyword search.
        """
        if chunks_path in self._inverted_cache:
            return
        chunks = self._load_chunks(chunks_path)
        inv = {}
        for i, c in enumerate(chunks):
            tokens = {t for t in re.findall(r"\w{3,}", c.text.lower())}
            for t in tokens:
                inv.setdefault(t, []).append(i)
        self._inverted_cache[chunks_path] = inv

    # ---------- FAISS helpers ----------
    def _load_faiss_if_present(self, index_path: str, chunks_path: str) -> Optional[faiss.Index]:
        if not index_path or not os.path.exists(index_path):
            return None
        if index_path in self._faiss_index_cache:
            return self._faiss_index_cache[index_path]
        try:
            idx = faiss.read_index(index_path)
            self._faiss_index_cache[index_path] = idx
            return idx
        except Exception as e:
            logger.warning(f"Failed to load FAISS index {index_path}: {e}")
            return None

    # ---------- public methods ----------
    def query_structured_chunks(
        self,
        query: str,
        index_path: str,
        chunks_path: str,
        section_filter: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        chunks = self._load_chunks(chunks_path)
        if not chunks:
            return []

        print("Used query:", query)
        faiss_idx = self._load_faiss_if_present(index_path, chunks_path)

        # Use FAISS if possible
        if faiss_idx is not None:
            q_emb = np.array(self.embedder.embed_text(query), dtype="float32").reshape(1, -1)
            # normalize
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            search_k = min(len(chunks), max(top_k * 3, top_k))
            try:
                scores, ids = faiss_idx.search(q_emb, search_k)
                return self._collect_ranked(chunks, scores[0], ids[0], section_filter, top_k, similarity_threshold)
            except Exception as e:
                logger.warning(f"FAISS search failed, falling back to in-memory: {e}")

        # fallback: in-memory ranking (use cached embeddings)
        self._ensure_embeddings(chunks_path)
        embs = self._embeddings_cache.get(chunks_path)
        q_emb = np.array(self.embedder.embed_text(query), dtype="float32")
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = (embs @ q_emb).astype(float)  # cosine since normalized
        ranked_idxs = list(reversed(sims.argsort()))  # high to low
        ranked = [(float(sims[i]), int(i)) for i in ranked_idxs]

        # convert to expected result list (with similarity_score)
        return self._collect_ranked(chunks, [s for s, _ in ranked], [i for _, i in ranked], section_filter, top_k, similarity_threshold)

    def _collect_ranked(self, chunks, scores, ids, section_filter, top_k, similarity_threshold):
        results = []
        seen_texts = set()
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(chunks):
                continue
            if score < similarity_threshold:
                continue
            c = chunks[idx]
            if section_filter and c.section_type not in section_filter:
                continue
            sig = (c.text or "")#[:200]
            if sig in seen_texts:
                continue
            seen_texts.add(sig)
            results.append({"id": c.id, "section_type": c.section_type, "title": c.title, "text": c.text, "similarity_score": float(score), **c.metadata})
            if len(results) >= top_k:
                break
        return results

    def query_keyword_chunks(self, chunks_path: str, keywords: List[str], top_k: int = 10) -> List[Dict]:
        chunks = self._load_chunks(chunks_path)
        if not chunks:
            return []
        # Build simple inverted index if missing
        self._build_inverted(chunks_path)
        inv = self._inverted_cache[chunks_path]
        lowered = [k.strip().lower() for k in keywords if k.strip()]
        hits = []
        freq_scores = {}
        for i, c in enumerate(chunks):
            txt = c.text.lower()
            hit_count = sum(txt.count(k) for k in lowered)
            if hit_count:
                hits.append(i)
                freq_scores[i] = hit_count
        # sort by frequency
        hits = sorted(hits, key=lambda i: freq_scores[i], reverse=True)
        results = []
        for i in hits[:top_k]:
            c = chunks[i]
            results.append({"id": c.id, "section_type": c.section_type, "title": c.title, "text": c.text, "similarity_score": 0.0, **c.metadata})
        return results

    def get_section_types(self, chunks_path: str) -> List[str]:
        chunks = self._load_chunks(chunks_path)
        return sorted({c.section_type for c in chunks})

