from __future__ import annotations

from episcope.rag.indexing.chunking import NoChunker, RecursiveChunker


def test_recursive_chunker_uses_inner_chunker() -> None:
    chunker = RecursiveChunker(max_chunk_size=5, inner_chunker=NoChunker())

    assert chunker.chunk("abcdefghij") == ["abcde", "fghij"]
    assert chunker.config["inner_chunker"]["strategy"] == "NoChunker"
