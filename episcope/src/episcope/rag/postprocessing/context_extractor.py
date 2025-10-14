import logging
from typing import Any, Dict, List, Sequence

from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.vectordb.base import AbstractVectorDB
from episcope.src.episcope.utils.types import SearchResult
from episcope.utils.processing_utils import TextProcessor

logger = logging.getLogger(__name__)

class ContextualSnippetRetriever(AbstractRetriever):
    """
    A retriever that extracts context snippets around availability terms
    from all chunks in a given namespace.
    """

    def __init__(self, vectordb: AbstractVectorDB, text_processor: TextProcessor):
        self.vectordb = vectordb
        self.text_processor = text_processor

    def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> Sequence[SearchResult]:
        """
        The query parameter is unused in this retriever.
        It scans all documents for the given paper_id.
        """
        paper_id = kwargs.get("paper_id")
        if not paper_id:
            raise ValueError("paper_id must be provided for context extraction.")

        try:
            chunks = self.vectordb.get_points(namespace=paper_id)
        except Exception as e:
            logger.debug(f"Failed to read chunks for context extraction for paper {paper_id}: {e}")
            return []

        results = []
        snippet_id = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            normalized = self.text_processor.normalize_text(text)
            sentences = self.text_processor.SENTENCE_SPLIT.split(normalized)

            for sent_idx, sentence in enumerate(sentences):
                artifacts = self.text_processor.extract_artifacts(sentence)

                if artifacts["availability_score"] > 0 or artifacts["has_artifacts"]:
                    context_sentences = []
                    if sent_idx > 0:
                        context_sentences.append(sentences[sent_idx - 1])
                    context_sentences.append(sentence)
                    if sent_idx < len(sentences) - 1:
                        context_sentences.append(sentences[sent_idx + 1])

                    context_text = " ".join(context_sentences).strip()

                    result = SearchResult(
                        id=f"context_{snippet_id}",
                        text=context_text,
                        section_type=chunk.get("section_type", "other"),
                        title=chunk.get("title", ""),
                        similarity_score=0.0,
                        source="context",
                        artifacts=artifacts
                    )
                    results.append(result)
                    snippet_id += 1
        
        return results[:top_k]
