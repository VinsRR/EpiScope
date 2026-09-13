from __future__ import annotations


def test_public_workflow_imports() -> None:
    from epilens.rag.retrieval import BaseRetriever, Retriever
    from epilens.services import EpiLensRuntime, RuntimeConfig
    from epilens.workflows.classification import (
        ClassificationDecision,
        DetailedClassificationResult,
        PaperClassifier,
    )
    from epilens.workflows.precision_miner import (
        DetailedExtractionResult,
        PrecisionMiner,
    )

    assert BaseRetriever is not None
    assert Retriever is not None
    assert EpiLensRuntime is not None
    assert RuntimeConfig is not None
    assert PaperClassifier is not None
    assert ClassificationDecision is not None
    assert DetailedClassificationResult is not None
    assert PrecisionMiner is not None
    assert DetailedExtractionResult is not None


def test_top_level_reexports() -> None:
    import epilens
    from epilens.workflows import PaperClassifier, PrecisionMiner

    # The headline workflow classes are importable straight from `epilens`.
    assert epilens.PaperClassifier is PaperClassifier
    assert epilens.PrecisionMiner is PrecisionMiner
