from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from episcope.workflows.classification.output import DetailedClassificationResult

from .repository import TrainingRepository
from .schemas import CapturedTraceRecord


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


class TrainingCaptureSink:
    """Capture completed classifier traces into a repository."""

    def __init__(
        self,
        repository: TrainingRepository,
        *,
        classifier_kind: str,
        strategy_name: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.repository = repository
        self.classifier_kind = classifier_kind
        self.strategy_name = strategy_name
        self.config_snapshot = dict(config_snapshot or {})

    def make_record(
        self,
        *,
        paper_id: str,
        detailed_result: DetailedClassificationResult,
        gold_label: Optional[Any] = None,
    ) -> CapturedTraceRecord:
        return CapturedTraceRecord(
            record_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            classifier_kind=self.classifier_kind,
            paper_id=paper_id,
            strategy_name=self.strategy_name,
            config_snapshot=_json_ready(self.config_snapshot),
            gold_label=_json_ready(gold_label),
            decision=detailed_result.decision,
            provenance=detailed_result.provenance,
            trace=detailed_result.trace,
            training=detailed_result.training,
        )

    def capture(
        self,
        *,
        paper_id: str,
        detailed_result: DetailedClassificationResult,
        gold_label: Optional[Any] = None,
    ) -> CapturedTraceRecord:
        record = self.make_record(
            paper_id=paper_id,
            detailed_result=detailed_result,
            gold_label=gold_label,
        )
        self.repository.save_raw(record)
        return record
