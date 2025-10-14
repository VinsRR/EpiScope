"""
A file-based vector database for storing paper-specific indexes.
"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .base import AbstractVectorDB

class FileDB(AbstractVectorDB):
    """A file-based vector database for storing paper-specific indexes."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._models: Dict[str, str] = {}
        self._load_existing_indices()

    def _load_existing_indices(self):
        for ns_dir in self.index_dir.iterdir():
            if ns_dir.is_dir():
                namespace = ns_dir.name
                self.load(namespace)

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None, embed_model: Optional[str] = None) -> None:
        if not namespace:
            raise ValueError("A namespace (paper_id) must be provided for PaperFileDB.")

        if embed_model:
            self._models[namespace] = embed_model

        points_list = list(points)
        if not points_list:
            if embed_model: # Save model even if no points
                self.save(namespace)
            return

        if namespace not in self._metadata:
            self.load(namespace)

        existing_points = {
            meta.get("id"): {"vector": vec, "payload": meta}
            for vec, meta in zip(self._embeddings.get(namespace, []), self._metadata.get(namespace, []))
        }

        for p in points_list:
            point_id = p["payload"].get("id")
            existing_points[point_id] = p

        updated_points = list(existing_points.values())
        if not updated_points:
            return

        self._embeddings[namespace] = np.array([p["vector"] for p in updated_points], dtype="float32")
        self._metadata[namespace] = [p["payload"] for p in updated_points]
        self.save(namespace)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        if not namespace:
            raise ValueError("A namespace (paper_id) must be provided for PaperFileDB.")

        if namespace not in self._embeddings:
            if not self.load(namespace):
                return []

        embs = self._embeddings[namespace]
        metadata = self._metadata[namespace]
        
        if filter:
            indices_to_search = [
                i for i, meta in enumerate(metadata)
                if all(meta.get(key) == value for key, value in filter.items())
            ]
            if not indices_to_search:
                return []
            embs = embs[indices_to_search]
            metadata = [metadata[i] for i in indices_to_search]

        if embs.shape[0] == 0:
            return []

        q_emb = np.array(query_vector, dtype="float32")
        sims = (embs @ q_emb).astype(float)
        
        actual_top_k = min(top_k, len(sims))
        top_idx_local = sims.argsort()[-actual_top_k:][::-1]

        results = []
        for idx_local in top_idx_local:
            meta = metadata[idx_local]
            results.append({**meta, "score": float(sims[idx_local])})
        return results

    def get_points(self, namespace: str, filter: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        if namespace not in self._metadata:
            if not self.load(namespace):
                return []
        
        points = self._metadata.get(namespace, [])
        
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
        paper_dir = self.index_dir / namespace
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        if namespace in self._embeddings and self._embeddings[namespace].size > 0:
            np.save(paper_dir / "embeddings.npy", self._embeddings[namespace])
        
        if namespace in self._metadata:
            with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata[namespace], f, indent=2)

        if namespace in self._models:
            with open(paper_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"embed_model": self._models[namespace]}, f, indent=2)

    def load(self, namespace: str) -> bool:
        paper_dir = self.index_dir / namespace
        if not paper_dir.exists():
            return False
        
        try:
            if (paper_dir / "embeddings.npy").exists():
                self._embeddings[namespace] = np.load(paper_dir / "embeddings.npy")
            if (paper_dir / "metadata.json").exists():
                with open(paper_dir / "metadata.json", "r", encoding="utf-8") as f:
                    self._metadata[namespace] = json.load(f)
            if (paper_dir / "config.json").exists():
                with open(paper_dir / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self._models[namespace] = config.get("embed_model")
            return True
        except Exception:
            return False
