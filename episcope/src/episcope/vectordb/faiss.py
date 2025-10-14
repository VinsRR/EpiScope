"""
A file-based vector database using FAISS.
"""
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
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._faiss_indices: Dict[str, faiss.Index] = {}
        self._models: Dict[str, str] = {}
        self._load_existing_indices()

    def _load_existing_indices(self):
        for ns_dir in self.index_dir.iterdir():
            if ns_dir.is_dir():
                namespace = ns_dir.name
                self.load(namespace)

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None) -> None:
        ns = namespace or "default"
        
        if embed_model:
            self._models[ns] = embed_model

        points_list = list(points)
        if not points_list:
            if embed_model:
                self.save(ns)
            return

        if ns not in self._metadata:
            self.load(ns)

        existing_meta = self._metadata.setdefault(ns, [])
        existing_embeds_list = self._embeddings.get(ns, np.array([])).tolist()

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
            self._embeddings[ns] = all_embeddings
            index = faiss.IndexFlatIP(all_embeddings.shape[1])
            index.add(all_embeddings)
            self._faiss_indices[ns] = index
        
        self.save(ns)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        ns = namespace or "default"
        if ns not in self._faiss_indices:
            if not self.load(ns):
                return []

        q_emb = np.array(query_vector, dtype="float32").reshape(1, -1)
        index = self._faiss_indices[ns]

        search_k = top_k
        if filter:
            search_k = min(index.ntotal, max(top_k * 5, 20))

        distances, indices = index.search(q_emb, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata[ns]):
                continue
            
            meta = self._metadata[ns][idx]

            if filter and not all(meta.get(key) == value for key, value in filter.items()):
                continue
            
            results.append({**meta, "score": float(dist)})
            
            if len(results) >= top_k:
                break
                
        return results

    def get_points(self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        ns = namespace or "default"
        if ns not in self._metadata:
            if not self.load(ns):
                return []
        
        points = self._metadata.get(ns, [])

        if filter:
            return [
                point for point in points
                if all(point.get(key) == value for key, value in filter.items())
            ]
            
        return points

    def get_embedding_model(self, namespace: str) -> Optional[str]:
        """Get the name of the embedding model used for a given namespace."""
        if namespace not in self._models:
            self.load(namespace)
        return self._models.get(namespace)

    def save(self, namespace: str) -> None:
        ns_dir = self.index_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        
        if namespace in self._metadata:
            with open(ns_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata[namespace], f, indent=2)
        
        if namespace in self._faiss_indices:
            index_path = str(ns_dir / "index.faiss")
            faiss.write_index(self._faiss_indices[namespace], index_path)

        if namespace in self._models:
            with open(ns_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"embed_model": self._models[namespace]}, f, indent=2)

    def load(self, namespace: str) -> bool:
        ns_dir = self.index_dir / namespace
        if not ns_dir.exists():
            return False
            
        try:
            if (ns_dir / "metadata.json").exists():
                with open(ns_dir / "metadata.json", "r", encoding="utf-8") as f:
                    self._metadata[namespace] = json.load(f)
            
            if (ns_dir / "index.faiss").exists():
                self._faiss_indices[namespace] = faiss.read_index(str(ns_dir / "index.faiss"))

            if (ns_dir / "config.json").exists():
                with open(ns_dir / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self._models[namespace] = config.get("embed_model")
            
            return True
        except Exception:
            return False
