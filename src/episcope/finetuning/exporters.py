from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .repository import TrainingRepository
from .schemas import CapturedTraceRecord, ReviewDecision, TaskBucket


def _assistant_completion(
    record: CapturedTraceRecord, review: ReviewDecision
) -> Optional[str]:
    if review.edited_completion:
        return review.edited_completion
    if record.training is not None and record.training.raw_llm_response is not None:
        return record.training.raw_llm_response
    if record.trace is not None:
        return record.trace.raw_llm_response
    return None


def _write_jsonl(samples: List[Dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")
    return path


class SFTExporter:
    """Export reviewed traces into simple SFT-ready JSONL datasets."""

    def __init__(self, repository: TrainingRepository) -> None:
        self.repository = repository

    def export_json_validity(self, output_path: str | Path) -> Path:
        samples: List[Dict[str, Any]] = []
        for record, review in self.repository.iter_approved(
            bucket=TaskBucket.JSON_VALIDITY
        ):
            sample = self._json_validity_sample(record, review)
            if sample is not None:
                samples.append(sample)
        return _write_jsonl(samples, output_path)

    def export_classification_core(self, output_path: str | Path) -> Path:
        samples: List[Dict[str, Any]] = []
        for record, review in self.repository.iter_approved(
            bucket=TaskBucket.CLASSIFICATION_CORE
        ):
            sample = self._classification_core_sample(record, review)
            if sample is not None:
                samples.append(sample)
        return _write_jsonl(samples, output_path)

    def _json_validity_sample(
        self,
        record: CapturedTraceRecord,
        review: ReviewDecision,
    ) -> Optional[Dict[str, Any]]:
        if record.training is None:
            return None
        completion = _assistant_completion(record, review)
        if completion is None or not record.training.prompt_messages:
            return None
        return {
            "messages": [
                *record.training.prompt_messages,
                {"role": "assistant", "content": completion},
            ],
            "paper_id": record.paper_id,
            "record_id": record.record_id,
            "task_bucket": TaskBucket.JSON_VALIDITY.value,
            "classifier_kind": record.classifier_kind,
        }

    def _classification_core_sample(
        self,
        record: CapturedTraceRecord,
        review: ReviewDecision,
    ) -> Optional[Dict[str, Any]]:
        if record.training is None:
            return None
        completion = _assistant_completion(record, review)
        if completion is None or not record.training.prompt_messages:
            return None
        return {
            "messages": [
                *record.training.prompt_messages,
                {"role": "assistant", "content": completion},
            ],
            "paper_id": record.paper_id,
            "record_id": record.record_id,
            "task_bucket": TaskBucket.CLASSIFICATION_CORE.value,
            "classifier_kind": record.classifier_kind,
            "updated_label": review.updated_label,
        }
