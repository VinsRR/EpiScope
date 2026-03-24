import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np

from .base import AbstractVectorDB

class FaissDB(AbstractVectorDB):
    """
    A file-based vector database using FAISS.

    This implementation keeps a separate copy of the embeddings (`_embeddings`) alongside
    the FAISS index (`_faiss_index`). The `_embeddings` array serves as the source of
    truth, allowing for easier updates (upserts) by rebuilding the index from scratch.
    The `_faiss_index` is treated as a performance cache for efficient searching.
    This design simplifies the upsert logic at the cost of performance for frequent updates.
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: np.ndarray = np.array([])
        self._metadata: List[Dict[str, Any]] = []
        self._faiss_index: Optional[faiss.Index] = None
        self._model: Optional[str] = None
        self._chunking_config: Optional[Dict[str, Any]] = None
        self._payload_keys: set[str] = set()
        self._loaded = False
        self._dirty = False
        self._load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def _load(self):
        try:
            if (self.index_dir / "metadata.json").exists():
                with open(self.index_dir / "metadata.json", "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            
            if (self.index_dir / "embeddings.npy").exists():
                self._embeddings = np.load(self.index_dir / "embeddings.npy")

            if (self.index_dir / "index.faiss").exists():
                self._faiss_index = faiss.read_index(str(self.index_dir / "index.faiss"))

            if (self.index_dir / "config.json").exists():
                with open(self.index_dir / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self._model = config.get("embed_model")
                    self._chunking_config = config.get("chunking_config")
                    payload_keys = config.get("payload_keys")
                    if payload_keys is not None:
                        self._payload_keys = set(payload_keys)
                self._loaded = True
        except Exception:
            # Silently fail if loading fails, will start with an empty DB
            pass

    def upsert(
        self,
        points: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
        embed_models: Optional[Dict[str, str]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        dense_model = embed_models.get("dense") if embed_models else None
        if dense_model:
            # Only set the DB-level model if it's not already configured.
            if self._model is None:
                self._model = dense_model
                self._dirty = True
            elif self._model != dense_model:
                assert False, (
                    f"Warning: upsert called with dense embed model={dense_model} "
                    f"but DB already has embed_model={self._model}; DB model not changed"
                )

        if chunking_config:
            if self._chunking_config is None:
                self._chunking_config = chunking_config
                self._dirty = True
            elif self._chunking_config != chunking_config:
                raise ValueError(f"Inconsistent chunking config. DB uses '{self._chunking_config}', but upsert was called with '{chunking_config}'.")

        points_list = list(points)
        if not points_list:
            return

        if not self._loaded:  # if this is the first instantiation create the payload keys
            for p in points_list:
                payload = p.setdefault("payload", {})
                if namespace:
                    payload["paper_id"] = namespace
                self._payload_keys.update(payload.keys())

        existing_meta = self._metadata
        existing_embeds_list = self._embeddings.tolist() if self._embeddings.size > 0 else []

        id_to_index = {meta.get("id"): i for i, meta in enumerate(existing_meta)}

        for p in points_list:
            point_id = p["payload"].get("id")
            if point_id in id_to_index:
                idx = id_to_index[point_id]
                existing_embeds_list[idx] = p["vector"]
                existing_meta[idx] = p["payload"]
            else:
                existing_embeds_list.append(p["vector"])
                existing_meta.append(p["payload"])

        if existing_embeds_list:
            all_embeddings = np.array(existing_embeds_list, dtype="float32")
            if all_embeddings.shape[0] > 0:
                self._embeddings = all_embeddings
                self._metadata = existing_meta
                index = faiss.IndexFlatIP(all_embeddings.shape[1])
                index.add(all_embeddings)
                self._faiss_index = index
        
        self._dirty = True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        combined_filter = dict(filter or {})
        if namespace:
            combined_filter["paper_id"] = namespace

        q_emb = np.array(query_vector, dtype="float32").reshape(1, -1)
        
        search_k = top_k
        if combined_filter:
            search_k = min(self._faiss_index.ntotal, max(top_k * 5, 100))

        distances, indices = self._faiss_index.search(q_emb, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            
            meta = self._metadata[idx]

            if combined_filter and not all(meta.get(key) == value for key, value in combined_filter.items()):
                continue
            
            results.append({**meta, "score": float(dist)})
            
            if len(results) >= top_k:
                break
                
        return results

    def get_points(self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        combined_filter = dict(filter or {})
        if namespace:
            combined_filter["paper_id"] = namespace
        
        if not combined_filter:
            return self._metadata

        return [
            point for point in self._metadata
            if all(point.get(key) == value for key, value in combined_filter.items())
        ]

    def get_payload_keys(self) -> set[str]:
        """Get the set of all available payload keys."""
        return self._payload_keys

    def get_embedding_model(self) -> Optional[str]:
        """Get the name of the embedding model used for the database."""
        return self._model

    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        """Get the name of the chunking strategy used for the database."""
        return self._chunking_config

    def capabilities(self) -> Dict[str, bool]:
        return {
            "dense": True,
            "sparse": False,
            "late": False,
        }

    def save(self) -> None:
        if not self._dirty:
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)
        
        if self._embeddings.size > 0:
            np.save(self.index_dir / "embeddings.npy", self._embeddings)

        if self._faiss_index is not None:
            index_path = str(self.index_dir / "index.faiss")
            faiss.write_index(self._faiss_index, index_path)

        config = {
            "embed_model": self._model,
            "chunking_config": self._chunking_config,
            "payload_keys": list(self._payload_keys)
        }
        with open(self.index_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self._dirty = False
