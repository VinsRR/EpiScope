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


def test_training_record_export_helpers_and_reward() -> None:
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

    finetune = record.to_finetune_sample()
    rl_samples = record.to_rl_samples()
    dpo_pair = record.to_dpo_pair()

    assert record.reward == 1.0
    assert finetune is not None
    assert finetune["paper_id"] == "paper-1"
    assert finetune["messages"][-1]["role"] == "assistant"
    assert len(rl_samples) == 2
    assert rl_samples[0]["parsed_ok"] is True
    assert dpo_pair is not None
    assert dpo_pair["chosen"] == '{"classification":["A"]}'
    assert dpo_pair["rejected"] == "not json"


def test_training_record_handles_missing_prompt_or_gold_label() -> None:
    record = ClassificationTrainingRecord(
        paper_id="paper-2",
        result=_result(),
    )

    assert record.reward is None
    assert record.to_finetune_sample() is None
    assert record.to_rl_samples() == []
    assert record.to_dpo_pair() is None
