from __future__ import annotations

from pathlib import Path

import pytest

from episcope.vectordb.faiss import FaissDB
from episcope.vectordb.file import FileDB
from episcope.vectordb.qdrant import QdrantDB


def _point(point_id: str, text: str = "content"):
    return {
        "id": point_id,
        "payload": {"id": point_id, "text": text},
        "vectors": {"dense": [1.0, 0.0]},
    }


@pytest.mark.parametrize("db_cls", [FileDB, FaissDB])
def test_local_vectordbs_preserve_namespace_after_reload(
    tmp_path: Path,
    db_cls,
) -> None:
    index_dir = tmp_path / db_cls.__name__.lower()

    db = db_cls(str(index_dir))
    db.upsert([_point("a-1")], namespace="paper-a", embed_models={"dense": "fake"})
    db.save()

    reopened = db_cls(str(index_dir))
    reopened.upsert([_point("b-1")], namespace="paper-b", embed_models={"dense": "fake"})

    assert reopened.get_points(namespace="paper-a") == [
        {"id": "a-1", "text": "content", "paper_id": "paper-a"}
    ]
    assert reopened.get_points(namespace="paper-b") == [
        {"id": "b-1", "text": "content", "paper_id": "paper-b"}
    ]


def test_qdrant_hybrid_guard_checks_capability_methods() -> None:
    db = object.__new__(QdrantDB)
    db.use_dense = True
    db.use_sparse = False

    with pytest.raises(ValueError, match="Hybrid search requires both dense and sparse"):
        db.search_hybrid(
            dense_query=[1.0],
            sparse_query={"indices": [], "values": []},
            top_k=1,
            prefetch_k=1,
        )
