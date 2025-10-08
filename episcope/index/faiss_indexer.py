"""FAISS‑based indexer implementation.

This module provides a simple implementation of the :class:`AbstractIndexer`
using the FAISS library for vector storage and similarity search.  When
FAISS is unavailable or no GPU/accelerator is present, the indexer
falls back to an in‑memory numpy dot product search.  Text
embeddings are computed via the :class:`SimplifiedEmbedder` from the
``episcope.retrieve.embeddings`` module, which itself gracefully
degrades if transformer models are not installed.

The indexer supports logical namespaces (e.g. per‑project or per‑paper)
so that multiple indices can coexist within the same object.  Each
namespace has its own embedding matrix and metadata store.  The
``index_documents`` method accepts an iterable of documents (dicts
containing at least a ``content`` field) and optionally a namespace.
The ``search`` method takes a query string and returns the top
matching documents from the specified namespace.
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Iterable, Sequence, Dict, Any, Optional, List

import numpy as np

from ..core.interfaces import AbstractIndexer
from ..retrieve.embeddings import SimplifiedEmbedder

logger = logging.getLogger(__name__)


try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class FaissIndexer(AbstractIndexer):
    """Vector indexer backed by FAISS or a numpy fallback.

    This class maintains a mapping of namespaces to FAISS indices
    together with the corresponding document metadata.  If FAISS is
    not available, the ``search`` method uses a brute force dot
    product computed with numpy.
    """

    def __init__(
        self,
        embed_model: str = "distilbert-base-uncased",
        batch_size: int = 8,
        *,
        index_dir: Optional[str | Path] = None,
    ) -> None:
        self.embedder = SimplifiedEmbedder(embed_model=embed_model, batch_size=batch_size)
        self.index_dir = Path(index_dir) if index_dir else None
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
        # namespace -> (embeddings matrix (numpy), list of metadata dicts)
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        # namespace -> faiss.Index
        self._faiss_indices: Dict[str, faiss.Index] = {} if _HAS_FAISS else {}

    def index_documents(self, docs: Iterable[Dict[str, Any]], *, namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        docs_list = list(docs)
        if not docs_list:
            return
        # Ensure there is a metadata list for this namespace
        meta_list = self._metadata.setdefault(ns, [])
        # Extract contents for embedding
        texts = [doc.get("content", doc.get("text", "")) for doc in docs_list]
        embeddings = np.array(self.embedder.embed_texts(texts), dtype="float32")
        # Normalize embeddings for cosine similarity
        if embeddings.size > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            embeddings = embeddings / norms
        # Append to existing embeddings
        if ns not in self._embeddings:
            self._embeddings[ns] = embeddings
        else:
            self._embeddings[ns] = np.vstack([self._embeddings[ns], embeddings])
        # Append metadata
        for doc in docs_list:
            meta_list.append(doc)
        self._metadata[ns] = meta_list
        # Rebuild or extend FAISS index if available.
        if _HAS_FAISS:
            try:
                if ns not in self._faiss_indices:
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    self._faiss_indices[ns] = index
                else:
                    index = self._faiss_indices[ns]
                index.add(embeddings)
                logger.debug(f"Added {len(docs_list)} embeddings to FAISS index for namespace '{ns}'")
            except Exception as e:
                logger.warning(f"Failed to build FAISS index; falling back to numpy: {e}")
        self.save(ns)

    def save(self, namespace: str) -> None:
        """Save the index for a given namespace to disk."""
        if not self.index_dir:
            return
        ns_dir = self.index_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        # Save metadata
        with open(ns_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata[namespace], f, indent=2)
        # Save FAISS index
        if _HAS_FAISS and namespace in self._faiss_indices:
            faiss.write_index(self._faiss_indices[namespace], str(ns_dir / "index.faiss"))
        else:
            # Save numpy embeddings as a fallback
            np.save(ns_dir / "embeddings.npy", self._embeddings[namespace])
        logger.debug(f"Saved index for namespace {namespace} to {ns_dir}")

    def load(self, namespace: str) -> bool:
        """Load the index for a given namespace from disk."""
        if not self.index_dir:
            return False
        ns_dir = self.index_dir / namespace
        if not ns_dir.exists():
            return False
        try:
            # Load metadata
            with open(ns_dir / "metadata.json", "r", encoding="utf-8") as f:
                self._metadata[namespace] = json.load(f)
            # Load FAISS index
            if _HAS_FAISS and (ns_dir / "index.faiss").exists():
                self._faiss_indices[namespace] = faiss.read_index(str(ns_dir / "index.faiss"))
            else:
                # Load numpy embeddings as a fallback
                self._embeddings[namespace] = np.load(ns_dir / "embeddings.npy")
            logger.debug(f"Loaded index for namespace {namespace} from {ns_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load index for namespace {namespace}: {e}")
            return False

    def _search_numpy(self, query_emb: np.ndarray, ns: str, top_k: int) -> Sequence[Dict[str, Any]]:
        embs = self._embeddings.get(ns)
        if embs is None or embs.size == 0:
            return []
        # compute dot products (cosine since normalized)
        sims = (embs @ query_emb).astype(float)
        # get indices of top k results
        top_ids = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_ids:
            meta = self._metadata[ns][idx]
            results.append({**meta, "score": float(sims[idx])})
        return results

    def search(self, query: str, *, top_k: int = 5, namespace: Optional[str] = None) -> Sequence[Dict[str, Any]]:
        ns = namespace or "default"
        # Load index from disk if not in memory
        if ns not in self._embeddings and ns not in self._faiss_indices:
            self.load(ns)
        # embed query and normalize
        q_emb = np.array(self.embedder.embed_text(query), dtype="float32")
        if q_emb.size > 0:
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        # use FAISS if available
        if _HAS_FAISS and ns in self._faiss_indices:
            index = self._faiss_indices[ns]
            try:
                distances, indices = index.search(q_emb.reshape(1, -1), top_k)
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._metadata[ns]):
                        continue
                    meta = self._metadata[ns][idx]
                    results.append({**meta, "score": float(dist)})
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed for namespace '{ns}'; falling back to numpy: {e}")
        # fallback to numpy search
        return self._search_numpy(q_emb, ns, top_k)
