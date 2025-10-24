import logging
from typing import Any, Dict, Optional, Sequence, Union

from episcope.clients import LLMClient, OllamaClient
from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.retrieval.hyde import HYDE
from episcope.vectordb.base import AbstractVectorDB
from episcope.schemas import SearchResult

logger = logging.getLogger(__name__)

class SemanticRetriever(AbstractRetriever):
    """Performs semantic search using a vector database, with optional HYDE."""

    def __init__(
        self,
        vectordb: AbstractVectorDB,
        hyde: Optional[Union[bool, HYDE]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.vectordb = vectordb
        self._allowed_filter_keys = self.vectordb.get_payload_keys()

        if hyde is True:
            # if not llm_client:
            #     raise ValueError("LLMClient must be provided when hyde is enabled.")
            self.hyde: Optional[HYDE] = HYDE(client=llm_client if llm_client else OllamaClient())
        elif isinstance(hyde, HYDE):
            self.hyde = hyde
        else:
            self.hyde = None

        model_name = self.vectordb.get_embedding_model()
        if not model_name:
            raise ValueError("VectorDB does not have an embedding model configured. Cannot perform semantic search.")

        logger.info(f"VectorDB is configured with embedding model: '{model_name}'. Instantiating corresponding embedder for retrieval.")
        self.embedder = EmbedderFactory.get_embedder(model_name)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """
        Perform semantic search for a single query.
        An optional 'filter' dictionary can be used for metadata filtering,
        including scoping to a 'paper_id'.
        """
        final_filter = filter.copy() if filter else {}
        namespace = final_filter.pop("paper_id", None) or final_filter.pop("namespace", None)

        for key in final_filter:
            if key not in self._allowed_filter_keys:
                raise ValueError(f"Invalid filter key: {key}. Allowed keys are: {self._allowed_filter_keys}")

        final_query = query
        if self.hyde:
            hypothetical_docs = self.hyde.generate_multiple(query, num_docs=1)
            valid_docs = [doc for doc in hypothetical_docs if doc and "failed" not in doc.lower()]
            if valid_docs:
                final_query = f"{query}\n\n" + "\n\n".join(valid_docs)
                logger.debug(f"HYDE enhanced query: {final_query}")

        try:
            query_embedding = self.embedder.embed_text(final_query)
            chunks = self.vectordb.search(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=final_filter if final_filter else None,
            )

            results = []
            for chunk in chunks:
                score = chunk.get("score", 0.0)
                if score < similarity_threshold:
                    continue

                results.append(
                    SearchResult(
                        id=str(chunk.get("id", "")),
                        paper_id=chunk.get("paper_id", ""),
                        text=chunk.get("text", ""),
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=score,
                        source="semantic",
                    )
                )
            return results

        except Exception as e:
            log_msg = f"Semantic search failed for query '{query}'"
            if namespace:
                log_msg += f" on namespace {namespace}"
            if final_filter:
                log_msg += f" with filter {final_filter}"
            log_msg += f": {e}"
            logger.debug(log_msg)
            return []

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        """
        Perform semantic search for a single query scoped to a specific paper.
        """
        final_filter = filter.copy() if filter else {}
        final_filter["paper_id"] = paper_id
        return self.retrieve(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=final_filter,
        )
