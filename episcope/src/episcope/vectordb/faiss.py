import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np

from .base import AbstractVectorDB

class FaissDB(AbstractVectorDB):
    """A file-based vector database using FAISS."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: np.ndarray = np.array([])
        self._metadata: List[Dict[str, Any]] = []
        self._faiss_index: Optional[faiss.Index] = None
        self._model: Optional[str] = None
        self._payload_keys: set[str] = set()
        self._loaded = False
        self._load()

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
            
            if (self.index_dir / "payload_keys.json").exists():
                with open(self.index_dir / "payload_keys.json", "r", encoding="utf-8") as f:
                    self._payload_keys = set(json.load(f))
                self._loaded = True
        except Exception:
            # Silently fail if loading fails, will start with an empty DB
            pass

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None) -> None:
        if embed_model:
            self._model = embed_model

        points_list = list(points)
        if not points_list:
            if embed_model:
                self.save()
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
        
        self.save()

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

    def get_points(self, namespace: str, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        combined_filter = dict(filter or {})
        combined_filter["paper_id"] = namespace
        
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

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)
        
        if self._embeddings.size > 0:
            np.save(self.index_dir / "embeddings.npy", self._embeddings)

        if self._faiss_index is not None:
            index_path = str(self.index_dir / "index.faiss")
            faiss.write_index(self._faiss_index, index_path)

        if self._model:
            with open(self.index_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"embed_model": self._model}, f, indent=2)
        
        with open(self.index_dir / "payload_keys.json", "w", encoding="utf-8") as f:
            json.dump(list(self._payload_keys), f, indent=2)
