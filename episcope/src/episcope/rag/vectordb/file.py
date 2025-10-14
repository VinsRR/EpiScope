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
        self._load_existing_indices()

    def _load_existing_indices(self):
        for ns_dir in self.index_dir.iterdir():
            if ns_dir.is_dir():
                namespace = ns_dir.name
                self.load(namespace)

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        if not namespace:
            raise ValueError("A namespace (paper_id) must be provided for PaperFileDB.")

        points_list = list(points)
        if not points_list:
            return

        # Load existing data if not in memory
        if namespace not in self._metadata:
            self.load(namespace)

        # Create a map of existing IDs to their data
        existing_points = {
            meta.get("id"): {"vector": vec, "payload": meta}
            for vec, meta in zip(self._embeddings.get(namespace, []), self._metadata.get(namespace, []))
        }

        # Update with new points
        for p in points_list:
            point_id = p["payload"].get("id")
            existing_points[point_id] = p

        # Reconstruct embeddings and metadata
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
    ) -> Sequence[Dict[str, Any]]:
        if not namespace:
            raise ValueError("A namespace (paper_id) must be provided for PaperFileDB.")

        if namespace not in self._embeddings:
            if not self.load(namespace):
                return []

        embs = self._embeddings[namespace]
        q_emb = np.array(query_vector, dtype="float32")
        
        sims = (embs @ q_emb).astype(float)
        top_idx = sims.argsort()[-top_k:][::-1]

        results = []
        for idx in top_idx:
            meta = self._metadata[namespace][idx]
            results.append({**meta, "score": float(sims[idx])})
        return results

    def save(self, namespace: str) -> None:
        paper_dir = self.index_dir / namespace
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(paper_dir / "embeddings.npy", self._embeddings[namespace])
        with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata[namespace], f, indent=2)

    def load(self, namespace: str) -> bool:
        paper_dir = self.index_dir / namespace
        if not paper_dir.exists():
            return False
            
        self._embeddings[namespace] = np.load(paper_dir / "embeddings.npy")
        with open(paper_dir / "metadata.json", "r", encoding="utf-8") as f:
            self._metadata[namespace] = json.load(f)
        return True
