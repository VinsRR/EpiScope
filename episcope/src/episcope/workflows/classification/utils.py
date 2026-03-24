from __future__ import annotations

from typing import Dict, TypeAlias

from episcope.schemas import SearchResult

PromptMessage: TypeAlias = Dict[str, str]


def result_score(chunk: SearchResult) -> float:
    """Return the ranking score used throughout classification retrieval."""
    if getattr(chunk, "rank_score", None) is not None:
        return chunk.rank_score
    if getattr(chunk, "similarity_score", None) is not None:
        return chunk.similarity_score
    return 0.0
