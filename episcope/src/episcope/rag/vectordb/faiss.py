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
        self._embeddings: Dict[str, np.ndarray] = {} # are these even needed?
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._faiss_indices: Dict[str, faiss.Index] = {}
        self._load_existing_indices()

    def _load_existing_indices(self):
        for ns_dir in self.index_dir.iterdir():
            if ns_dir.is_dir():
                namespace = ns_dir.name
                self.load(namespace)

    def upsert(self, points: Iterable[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        points_list = list(points)
        if not points_list:
            return

        # Load existing data if not in memory
        if ns not in self._metadata:
            self.load(ns)

        existing_meta = self._metadata.setdefault(ns, [])
        existing_embeds = self._embeddings.get(ns)

        # Create a map of existing IDs to their index
        id_to_index = {meta.get("id"): i for i, meta in enumerate(existing_meta)}

        new_vectors = []
        new_meta = []

        for p in points_list:
            point_id = p["payload"].get("id")
            if point_id in id_to_index:
                # Update existing point
                idx = id_to_index[point_id]
                existing_embeds[idx] = np.array(p["vector"], dtype="float32")
                existing_meta[idx] = p["payload"]
            else:
                # Add as a new point
                new_vectors.append(p["vector"])
                new_meta.append(p["payload"])

        if new_vectors:
            new_embeddings = np.array(new_vectors, dtype="float32")
            if existing_embeds is not None and existing_embeds.size > 0:
                self._embeddings[ns] = np.vstack([existing_embeds, new_embeddings])
            else:
                self._embeddings[ns] = new_embeddings
            self._metadata[ns].extend(new_meta)

        # Rebuild the FAISS index
        all_embeddings = self._embeddings[ns]
        if all_embeddings.size > 0:
            index = faiss.IndexFlatIP(all_embeddings.shape[1])
            index.add(all_embeddings)
            self._faiss_indices[ns] = index
        
        self.save(ns)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> Sequence[Dict[str, Any]]:
        ns = namespace or "default"
        if ns not in self._faiss_indices:
            self.load(ns)

        if ns not in self._faiss_indices:
            return []

        q_emb = np.array(query_vector, dtype="float32").reshape(1, -1)
        index = self._faiss_indices[ns]
        distances, indices = index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata[ns]):
                continue
            meta = self._metadata[ns][idx]
            results.append({**meta, "score": float(dist)})
        return results

    def save(self, namespace: str) -> None:
        ns_dir = self.index_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        
        with open(ns_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata[namespace], f, indent=2)
            
        index_path = str(ns_dir / "index.faiss")
        faiss.write_index(self._faiss_indices[namespace], index_path)

    def load(self, namespace: str) -> bool:
        ns_dir = self.index_dir / namespace
        if not ns_dir.exists():
            return False
            
        with open(ns_dir / "metadata.json", "r", encoding="utf-8") as f:
            self._metadata[namespace] = json.load(f)
            
        self._faiss_indices[namespace] = faiss.read_index(str(ns_dir / "index.faiss"))
        return True
