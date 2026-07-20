"""Chunking utilities for indexing."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from episcope.rag.embeddings.base import Embedder
from episcope.schemas import PaperMetadata, StructuredSection


class Chunker(ABC):
    """Abstract base class for a chunker."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Chunk a single text."""
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        """
        JSON-serializable parameters describing this chunker instance.
        Other code can persist/compare this dict.
        """
        base: Dict[str, Any] = {"strategy": self.__class__.__name__}
        base.update(self._config())
        return base

    def _config(self) -> Dict[str, Any]:
        """Subclass hook: return only the parameters specific to the strategy."""
        return {}


def _split_into_sentences(text: str) -> List[str]:
    """Very light sentence splitter (period/exclamation/question)."""
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_endings.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_paper(
    chunker: Chunker,
    sections: List[StructuredSection],
    metadata: PaperMetadata,
    paper_id: str,
) -> List[Dict[str, Any]]:
    """Create chunk documents from a paper using a specific chunker."""
    chunks: List[Dict[str, Any]] = []

    # Metadata chunking (abstract)
    if metadata and getattr(metadata, "abstract", None):
        text = metadata.abstract
        # The abstract is treated as a single chunk, not split further by the chunker.
        chunks.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
                "text": text,
                "content": text,
                "section_title": "Abstract",
                "section_type": "Abstract",
                "paper_id": paper_id,
                "is_metadata": True,
            }
        )

    # Section chunking
    for section in sections:
        section_texts = chunker.chunk(section.content)
        for text in section_texts:
            chunks.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
                    "text": text,
                    "content": text,
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "paper_id": paper_id,
                    "is_metadata": False,
                }
            )
    return chunks


class NoChunker(Chunker):
    """Treats each input text as a single chunk."""

    def chunk(self, text: str) -> List[str]:
        return [text]


class SentenceChunker(Chunker):
    """Produces sentence-level chunks for a structured section."""

    def chunk(self, text: str) -> List[str]:
        """Split text into sentences."""
        return _split_into_sentences(text)


class ParagraphChunker(Chunker):
    """Produces paragraph chunks for a structured section."""

    def __init__(self, min_chunk_size: int = 50):
        self.min_chunk_size = int(min_chunk_size)

    def _config(self) -> Dict[str, Any]:
        return {"min_chunk_size": self.min_chunk_size}

    def chunk(self, text: str) -> List[str]:
        """Split on blank-line paragraphs; drop tiny/very-short ones."""
        chunks: List[str] = []
        # Normalize paragraph breaks (two or more newlines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for paragraph in paragraphs:
            if len(paragraph) >= self.min_chunk_size and len(paragraph.split()) > 5:
                chunks.append(paragraph)
        return chunks


class FixedSizeChunker(Chunker):
    """Splits text into fixed-size character chunks with overlap."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            # Prevent infinite loops / zero or negative step
            self.chunk_overlap = self.chunk_size - 1

    def _config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            # Ensure forward progress
            start += max(1, step)
        return chunks


class RecursiveChunker(Chunker):
    """Recursively splits chunks until they are under a certain size."""

    def __init__(self, max_chunk_size: int = 1024, inner_chunker: Chunker = None):
        self._inner_chunker = inner_chunker or ParagraphChunker()
        self.max_chunk_size = int(max_chunk_size)
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")

    def _config(self) -> Dict[str, Any]:
        return {
            "max_chunk_size": self.max_chunk_size,
            "inner_chunker": self._inner_chunker.config,
        }

    def chunk(self, text: str) -> List[str]:
        initial_chunks = self._inner_chunker.chunk(text)
        final_chunks: List[str] = []
        for chunk_text in initial_chunks:
            if len(chunk_text) > self.max_chunk_size:
                final_chunks.extend(self._split(chunk_text))
            else:
                final_chunks.append(chunk_text)
        return final_chunks

    def _split(self, text: str) -> List[str]:
        sub_chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.max_chunk_size, n)
            sub_chunk_text = text[start:end].strip()
            if sub_chunk_text:
                sub_chunks.append(sub_chunk_text)
            start = end
        return sub_chunks


class SemanticChunker(Chunker):
    """Groups sentences into chunks based on semantic similarity (cosine).

    Two comparison strategies are supported:

    1) Compare-to-chunk (default): compare the next sentence's embedding to the
       embedding of the *current chunk as a whole* (running average of its sentence
       embeddings). This tends to preserve *global coherence* within each chunk and
       resists local drift.

    2) Compare-to-last: compare the next sentence only to the *immediately preceding*
       sentence. This is simpler/faster but can allow the chunk to drift gradually
       if the topic changes slowly over many sentences.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.5,
        compare_mode: Literal["chunk", "last"] = "chunk",
    ):
        """
        Args:
            embedder: Embedding model that exposes `embed_texts(List[str]) -> List[List[float]]`.
            similarity_threshold: Cosine similarity threshold to keep grouping.
            compare_mode:
                - "chunk": (Strategy 1, default) compare to the whole chunk embedding.
                - "last":  (Strategy 2) compare only to the last sentence embedding.
        """
        if embedder is None:
            from episcope.rag.embeddings.huggingface import HuggingFaceEmbedder

            embedder = HuggingFaceEmbedder()
        self.embedder = embedder
        self.similarity_threshold = float(similarity_threshold)
        self.compare_mode = compare_mode

    def _config(self) -> Dict[str, Any]:
        """Expose a stable, JSON-friendly configuration for storage/compare elsewhere."""
        embedder_id = (
            getattr(self.embedder, "model_name", None)
            or getattr(self.embedder, "name", None)
            or self.embedder.__class__.__name__
        )
        return {
            "similarity_threshold": float(self.similarity_threshold),
            "embedder": embedder_id,
            "compare_mode": self.compare_mode,
        }

    def chunk(self, text: str) -> List[str]:
        """Split `text` into semantically coherent chunks of sentences."""
        sentences = _split_into_sentences(text)
        if not sentences:
            return []

        # Embed sentences and L2-normalize to enable cosine via dot product.
        embeddings = self.embedder.embed_texts(sentences)
        embs = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms

        if self.compare_mode == "last":
            return self._chunk_by_compare_to_last(sentences, embs)
        # default: "chunk"
        return self._chunk_by_compare_to_chunk(sentences, embs)

    def _chunk_by_compare_to_chunk(
        self, sentences: List[str], embs: np.ndarray
    ) -> List[str]:
        """Strategy 1 (default): compare to the *whole chunk*.

        Useful when you want each chunk to remain *globally coherent*. We maintain
        a running (unnormalized) sum of sentence embeddings in the current chunk,
        compute its mean and re-normalize before comparing with the next sentence.
        This reduces topic drift across long chunks, at the cost of a tiny bit of
        extra computation per step.
        """
        chunks: List[str] = []
        current_sentences = [sentences[0]]

        # Running sum lets us update the chunk embedding in O(d) per step.
        running_sum = embs[0].copy()
        count = 1

        for i in range(1, len(sentences)):
            next_emb = embs[i]

            # Compute normalized mean embedding for the current chunk.
            chunk_mean = running_sum / max(1, count)
            chunk_mean = chunk_mean / (np.linalg.norm(chunk_mean) + 1e-9)

            similarity = float(np.dot(chunk_mean, next_emb))

            if similarity >= self.similarity_threshold:
                # Merge into current chunk
                current_sentences.append(sentences[i])
                running_sum += next_emb
                count += 1
            else:
                # Flush and start a new chunk
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]
                running_sum = next_emb.copy()
                count = 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _chunk_by_compare_to_last(
        self, sentences: List[str], embs: np.ndarray
    ) -> List[str]:
        """Strategy 2: compare to the *last sentence only*.

        Useful when you prioritize *simplicity and speed* and your documents are
        fairly homogeneous (or you accept gradual drift). We only compute cosine
        similarity between the new sentence and the immediately preceding one.
        """
        chunks: List[str] = []
        current_sentences = [sentences[0]]
        last_emb = embs[0]

        for i in range(1, len(sentences)):
            next_emb = embs[i]
            similarity = float(np.dot(last_emb, next_emb))

            if similarity >= self.similarity_threshold:
                current_sentences.append(sentences[i])
            else:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]

            # Allow drift by updating the reference to the most recent sentence.
            last_emb = next_emb

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks
