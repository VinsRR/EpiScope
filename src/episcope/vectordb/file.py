import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .base import AbstractVectorDB


class FileDB(AbstractVectorDB):
    """A file-based vector database for storing paper-specific indexes."""

    def __init__(self, index_dir: str, *, strict: bool = False):
        self.index_dir = Path(index_dir)
        self._strict = strict
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: np.ndarray = np.array([])
        self._metadata: List[Dict[str, Any]] = []
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
            if (self.index_dir / "embeddings.npy").exists():
                self._embeddings = np.load(self.index_dir / "embeddings.npy")
            if (self.index_dir / "metadata.json").exists():
                with open(self.index_dir / "metadata.json", "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            if (self.index_dir / "config.json").exists():
                with open(self.index_dir / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self._model = config.get("embed_model")
                    self._chunking_config = config.get("chunking_config")
                    payload_keys = config.get("payload_keys")
                    if payload_keys is not None:
                        self._payload_keys = set(payload_keys)
                self._loaded = True
            if self._strict:
                embedding_count = (
                    int(self._embeddings.shape[0]) if self._embeddings.size else 0
                )
                if self._embeddings.size and self._embeddings.ndim != 2:
                    raise ValueError("Stored embeddings must be a two-dimensional array.")
                if embedding_count != len(self._metadata):
                    raise ValueError(
                        "Stored metadata and embedding counts do not match."
                    )
                if self._metadata and not self._loaded:
                    raise ValueError("Stored index configuration is missing or unreadable.")
        except Exception:
            if self._strict:
                raise
            # Silently fail if loading fails, will start with an empty DB
            pass

    def upsert(
        self,
        points: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
        embed_models: Optional[Dict[str, str]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _dense_vector(point: Dict[str, Any]) -> List[float]:
            if "vector" in point:
                return point["vector"]
            vectors = point.get("vectors") or {}
            dense = vectors.get("dense")
            if dense is None:
                raise ValueError("Point is missing a dense vector.")
            return dense

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
                raise ValueError(
                    f"Inconsistent chunking config. DB uses '{self._chunking_config}', but upsert was called with '{chunking_config}'."
                )

        points_list = list(points)
        if not points_list:
            return

        for p in points_list:
            payload = p.setdefault("payload", {})
            if p.get("id") is not None:
                payload.setdefault("id", p["id"])
            if namespace:
                payload["paper_id"] = namespace
            self._payload_keys.update(payload.keys())

        existing_points = {
            meta.get("id"): {"vector": vec.tolist(), "payload": meta}
            for vec, meta in zip(self._embeddings, self._metadata)
        }

        for p in points_list:
            point_id = p["payload"].get("id")
            existing_points[point_id] = {
                "vector": _dense_vector(p),
                "payload": p["payload"],
            }

        updated_points = list(existing_points.values())
        if not updated_points:
            return

        self._embeddings = np.array(
            [p["vector"] for p in updated_points], dtype="float32"
        )
        self._metadata = [p["payload"] for p in updated_points]
        self._dirty = True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        combined_filter = dict(filter or {})
        if namespace:
            combined_filter["paper_id"] = namespace

        embs = self._embeddings
        metadata = self._metadata

        indices_to_search = list(range(len(metadata)))
        if combined_filter:
            indices_to_search = [
                i
                for i, meta in enumerate(metadata)
                if all(meta.get(key) == value for key, value in combined_filter.items())
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

    def get_points(
        self, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None
    ) -> Sequence[Dict[str, Any]]:
        """Retrieve points from a given namespace, with an optional filter."""
        combined_filter = dict(filter or {})
        if namespace:
            combined_filter["paper_id"] = namespace

        if not combined_filter:
            return self._metadata

        return [
            point
            for point in self._metadata
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

    def delete(self, namespace: str) -> int:
        keep = [
            index
            for index, metadata in enumerate(self._metadata)
            if metadata.get("paper_id") != namespace
        ]
        removed = len(self._metadata) - len(keep)
        if not removed:
            return 0
        if keep:
            self._embeddings = self._embeddings[keep]
            self._metadata = [self._metadata[index] for index in keep]
        else:
            self._embeddings = np.array([])
            self._metadata = []
        self._payload_keys = {
            key for metadata in self._metadata for key in metadata.keys()
        }
        self._dirty = True
        return removed

    def save(self) -> None:
        if not self._dirty:
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = self.index_dir / "embeddings.npy"
        if self._embeddings.size > 0:
            with tempfile.NamedTemporaryFile(
                dir=self.index_dir, suffix=".npy", delete=False
            ) as handle:
                np.save(handle, self._embeddings)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_embeddings = Path(handle.name)
            os.replace(temporary_embeddings, embeddings_path)
        elif embeddings_path.exists():
            embeddings_path.unlink()

        def _write_json_atomic(path: Path, value: Any) -> None:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.index_dir, suffix=".tmp", delete=False
            ) as handle:
                json.dump(value, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)

        _write_json_atomic(self.index_dir / "metadata.json", self._metadata)

        config = {
            "embed_model": self._model,
            "chunking_config": self._chunking_config,
            "payload_keys": list(self._payload_keys),
        }
        _write_json_atomic(self.index_dir / "config.json", config)

        self._dirty = False
