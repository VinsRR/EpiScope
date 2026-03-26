from __future__ import annotations


def test_public_workflow_imports() -> None:
    from episcope.rag.retrieval import BaseRetriever, Retriever
    from episcope.workflows.classification import (
        ClassificationDecision,
        DetailedClassificationResult,
        PaperClassifier,
    )
    from episcope.workflows.precision_miner import (
        DetailedExtractionResult,
        PrecisionMiner,
    )

    assert BaseRetriever is not None
    assert Retriever is not None
    assert PaperClassifier is not None
    assert ClassificationDecision is not None
    assert DetailedClassificationResult is not None
    assert PrecisionMiner is not None
    assert DetailedExtractionResult is not None
