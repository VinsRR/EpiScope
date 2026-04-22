from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from episcope.rag.retrieval.components import CandidateRetriever, QueryTransformer
from episcope.rag.retrieval.retriever import Retriever
from episcope.schemas import SearchResult
from episcope.vectordb.base import AbstractVectorDB


class _VectorDB(AbstractVectorDB):
    def __init__(self, capabilities: Dict[str, bool]) -> None:
        self._capabilities = capabilities

    def upsert(
        self,
        points: Iterable[Dict[str, Any]],
        namespace: Optional[str] = None,
        embed_models: Optional[Dict[str, str]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        return []

    def get_points(
        self,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Dict[str, Any]]:
        return []

    def get_payload_keys(self) -> set[str]:
        return {"year"}

    def get_embedding_model(self):
        return {"dense": "fake-dense", "sparse": "fake-sparse"}

    def get_chunking_config(self) -> Optional[Dict[str, Any]]:
        return None

    def capabilities(self) -> Dict[str, bool]:
        return dict(self._capabilities)


class _RecordingTransformer(QueryTransformer):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.calls: List[str] = []

    def transform(self, query: str) -> str:
        self.calls.append(query)
        return f"{query}{self.suffix}"


class _RecordingCandidateRetriever(CandidateRetriever):
    def __init__(self, source: str, results: List[Dict[str, Any]]) -> None:
        self.source = source
        self.results = list(results)
        self.calls: List[Dict[str, Any]] = []

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "namespace": namespace,
                "filter": filter,
            }
        )
        return self.results[:top_k]


def _candidate(doc_id: str, text: str, score: float) -> Dict[str, Any]:
    return {
        "id": doc_id,
        "paper_id": "paper-1",
        "text": text,
        "section_type": "Methods",
        "title": "",
        "score": score,
    }


def test_retriever_can_use_only_semantic_stage_even_if_db_supports_sparse() -> None:
    vectordb = _VectorDB({"dense": True, "sparse": True, "late": False})
    transformer = _RecordingTransformer("::expanded")
    semantic_stage = _RecordingCandidateRetriever(
        "semantic",
        [
            _candidate("1", "Dense result", 0.8),
            _candidate("2", "Second result", 0.6),
        ],
    )
    retriever = Retriever(
        vectordb=vectordb,
        candidate_retrievers=[semantic_stage],
        query_transformers=[transformer],
        use_rerank=False,
    )

    results = retriever.retrieve(
        "availability query",
        top_k=1,
        filter={"paper_id": "paper-1", "year": 2024},
    )

    assert transformer.calls == ["availability query"]
    assert semantic_stage.calls == [
        {
            "query": "availability query::expanded",
            "top_k": 1,
            "namespace": "paper-1",
            "filter": {"year": 2024},
        }
    ]
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "semantic"
    assert results[0].text == "Dense result"


def test_retriever_fuses_multiple_candidate_stages() -> None:
    vectordb = _VectorDB({"dense": True, "sparse": True, "late": False})
    dense_stage = _RecordingCandidateRetriever(
        "semantic",
        [
            _candidate("doc-1", "Dense first", 0.9),
            _candidate("shared", "Shared text", 0.8),
        ],
    )
    sparse_stage = _RecordingCandidateRetriever(
        "sparse",
        [
            _candidate("shared", "Shared text", 0.7),
            _candidate("doc-2", "Sparse second", 0.6),
        ],
    )
    retriever = Retriever(
        vectordb=vectordb,
        candidate_retrievers=[dense_stage, sparse_stage],
        use_rerank=False,
    )

    results = retriever.retrieve("hybrid query", top_k=3)

    assert [result.id for result in results] == ["shared", "doc-1", "doc-2"]
    assert all(result.source == "hybrid" for result in results)
