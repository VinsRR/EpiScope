from __future__ import annotations

import pytest

from episcope.rag.retrieval.fusion import RRFFusion


def test_rrf_fusion_combines_rank_lists_and_adds_rrf_score() -> None:
    fusion = RRFFusion(k=60)

    fused = fusion.fuse(
        [
            [
                {"id": "doc-1", "text": "A"},
                {"id": "doc-2", "text": "B"},
            ],
            [
                {"id": "doc-2", "text": "B"},
                {"id": "doc-3", "text": "C"},
            ],
        ],
        top_k=3,
    )

    assert [item["id"] for item in fused] == ["doc-2", "doc-1", "doc-3"]
    assert fused[0]["rrf_score"] == pytest.approx((1 / 62) + (1 / 61))
    assert "rrf_score" in fused[1]
