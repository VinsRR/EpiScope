from __future__ import annotations

from episcope.rag.embeddings.gemini import GeminiEmbedder
from episcope.rag.retrieval.candidates import SemanticCandidateRetriever


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts, *, model, output_dimensionality=None, task_type=None):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "output_dimensionality": output_dimensionality,
                "task_type": task_type,
            }
        )
        dim = output_dimensionality or 2
        return [[1.0] * dim for _ in texts]


class _QueryAwareEmbedder:
    model_name = "query-aware"

    def __init__(self) -> None:
        self.query_calls = []
        self.text_calls = []

    def embed_query(self, text: str):
        self.query_calls.append(text)
        return [1.0, 0.0]

    def embed_text(self, text: str):
        self.text_calls.append(text)
        return [0.0, 1.0]


class _DenseVectorDb:
    def get_payload_keys(self):
        return set()

    def get_embedding_model(self):
        return {"dense": "query-aware"}

    def search_dense(self, *, query_vector, top_k, namespace=None, filter=None):
        return [
            {
                "id": "chunk-1",
                "paper_id": "paper-1",
                "text": "Retrieved text",
                "score": query_vector[0],
            }
        ]


def test_gemini_embedder_uses_query_and_document_tasks() -> None:
    client = _FakeGeminiClient()
    embedder = GeminiEmbedder(client=client, fixed_dim=2)

    embedder.embed_query("where did the data come from?")
    embedder.embed_document("The data came from a public repository.")

    assert [call["task_type"] for call in client.calls] == [
        "RETRIEVAL_QUERY",
        "RETRIEVAL_DOCUMENT",
    ]


def test_semantic_candidate_retriever_prefers_embed_query() -> None:
    embedder = _QueryAwareEmbedder()
    retriever = SemanticCandidateRetriever(_DenseVectorDb(), dense_embedder=embedder)

    results = retriever.retrieve_candidates("data source", top_k=1)

    assert embedder.query_calls == ["data source"]
    assert embedder.text_calls == []
    assert results[0]["score"] == 1.0
