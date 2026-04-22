from __future__ import annotations

import json
from pathlib import Path

from episcope.finetuning import (
    SFTExporter,
    TaskBucket,
    TraceReviewer,
    TrainingCaptureSink,
    TrainingRepository,
)
from episcope.rag.provenance import Evidence, Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.output import (
    ClassificationDecision,
    ClassificationTrace,
    ClassificationTrainingRecord,
    CompletionSample,
    DetailedClassificationResult,
)
from episcope.workflows.classification.schemas import ClassificationResult, DataAccessibility


def _detailed_result() -> DetailedClassificationResult:
    result = ClassificationResult(
        classification=[DataAccessibility.OPEN],
        confidence=0.9,
        class_probabilities={"A": 0.9},
        evidence={"reasoning": "The dataset is on Zenodo."},
        extras={},
    )
    decision = ClassificationDecision(
        paper_id="paper-1",
        metadata=PaperMetadata(title="Paper", abstract="Abstract"),
        result=result,
        top_evidence=[
            SearchResult(
                id="1",
                paper_id="paper-1",
                text="Data are available on Zenodo.",
                section_type="Methods",
                similarity_score=0.8,
                rank_score=0.8,
                source="semantic",
                artifacts={"category": "open"},
            )
        ],
    )
    provenance = Provenance(
        answer='{"classification":["A"]}',
        evidences=[
            Evidence(
                paper_id="paper-1",
                snippet="Data are available on Zenodo.",
                section="open",
            )
        ],
    )
    trace = ClassificationTrace(
        prompt_messages=[{"role": "user", "content": "Classify this paper."}],
        raw_llm_response='{"classification":["A"]}',
    )
    training = ClassificationTrainingRecord(
        paper_id="paper-1",
        result=result,
        prompt_messages=[{"role": "user", "content": "Classify this paper."}],
        raw_llm_response='{"classification":["A"]}',
        all_samples=[
            CompletionSample(
                messages=[{"role": "user", "content": "Classify this paper."}],
                completion='{"classification":["A"]}',
                parsed_ok=True,
                reward=1.0,
            )
        ],
    )
    return DetailedClassificationResult(
        decision=decision,
        provenance=provenance,
        trace=trace,
        training=training,
    )


def test_capture_review_and_export_roundtrip(tmp_path: Path) -> None:
    repository = TrainingRepository(tmp_path)
    sink = TrainingCaptureSink(
        repository,
        classifier_kind="data_accessibility",
        strategy_name="grobid",
        config_snapshot={"top_k": 10},
    )

    record = sink.capture(paper_id="paper-1", detailed_result=_detailed_result())
    loaded = repository.load_raw(record.record_id)

    assert loaded.paper_id == "paper-1"
    assert loaded.classifier_kind == "data_accessibility"
    assert loaded.decision is not None
    assert loaded.decision.result.classification == ["OPEN"]

    reviewer = TraceReviewer(repository, reviewer="tester")
    packet = reviewer.get_review_packet(record.record_id)
    assert packet["summary"]["predicted_labels"] == ["OPEN"]

    review = reviewer.approve(
        record.record_id,
        buckets=[TaskBucket.JSON_VALIDITY, TaskBucket.CLASSIFICATION_CORE],
        notes="Looks good.",
    )
    assert review.buckets == [TaskBucket.JSON_VALIDITY, TaskBucket.CLASSIFICATION_CORE]

    exporter = SFTExporter(repository)
    json_validity_path = exporter.export_json_validity(tmp_path / "json_validity.jsonl")
    classification_core_path = exporter.export_classification_core(tmp_path / "classification_core.jsonl")

    json_validity_lines = json_validity_path.read_text(encoding="utf-8").strip().splitlines()
    classification_core_lines = classification_core_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(json_validity_lines) == 1
    assert len(classification_core_lines) == 1

    json_validity_sample = json.loads(json_validity_lines[0])
    classification_core_sample = json.loads(classification_core_lines[0])

    assert json_validity_sample["task_bucket"] == "json_validity"
    assert classification_core_sample["task_bucket"] == "classification_core"
    assert json_validity_sample["messages"][-1]["role"] == "assistant"
