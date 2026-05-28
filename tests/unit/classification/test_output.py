from __future__ import annotations

from episcope.workflows.classification.output import (
    ClassificationTrainingRecord,
    CompletionSample,
)
from episcope.workflows.classification.schemas import ClassificationResult, DataAccessibility


def _result() -> ClassificationResult:
    return ClassificationResult(
        classification=[DataAccessibility.OPEN],
        confidence=0.8,
        class_probabilities={"A": 0.8, "E": 0.2},
        evidence={"reasoning": "Open repository statement found."},
        extras={},
    )


def test_training_record_keeps_completion_samples_and_reward() -> None:
    record = ClassificationTrainingRecord(
        paper_id="paper-1",
        result=_result(),
        prompt_messages=[{"role": "user", "content": "Classify this paper."}],
        raw_llm_response='{"classification":["A"]}',
        all_samples=[
            CompletionSample(
                messages=[{"role": "user", "content": "Classify this paper."}],
                completion='{"classification":["A"]}',
                parsed_ok=True,
                reward=1.0,
            ),
            CompletionSample(
                messages=[{"role": "user", "content": "Classify this paper."}],
                completion="not json",
                parsed_ok=False,
                reward=0.0,
            ),
        ],
        gold_label=[DataAccessibility.OPEN],
    )

    assert record.reward == 1.0
    assert record.prompt_messages == [
        {"role": "user", "content": "Classify this paper."}
    ]
    assert record.raw_llm_response == '{"classification":["A"]}'
    assert len(record.all_samples) == 2
    assert record.all_samples[0].parsed_ok is True
    assert record.all_samples[1].completion == "not json"


def test_training_record_handles_missing_prompt_or_gold_label() -> None:
    record = ClassificationTrainingRecord(
        paper_id="paper-2",
        result=_result(),
    )

    assert record.reward is None
    assert record.prompt_messages == []
    assert record.raw_llm_response is None
    assert record.all_samples == []
