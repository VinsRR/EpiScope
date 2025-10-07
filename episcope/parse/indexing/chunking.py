"""Text chunking utilities for indexing.

This module provides strategies for splitting structured document sections
into smaller chunks suitable for embedding and indexing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..blueprints.data_blueprints import StructuredSection


def paragraph_chunking(section: StructuredSection, min_chunk_size: int) -> List[Dict[str, Any]]:
    """
    Produce paragraph chunks for a structured section.

    Chunks are created for paragraphs that exceed `min_chunk_size`
    characters and contain more than five words.
    """
    chunks: List[Dict[str, Any]] = []
    paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        if len(paragraph) > min_chunk_size and len(paragraph.split()) > 5:
            chunks.append({"text": paragraph})
    return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation heuristics.
    """
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    parts = sentence_endings.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

