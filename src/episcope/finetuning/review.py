from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .repository import TrainingRepository
from .schemas import CapturedTraceRecord, ReviewDecision, ReviewStatus, TaskBucket


def _label_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [getattr(item, "name", str(item)) for item in value]
    return [getattr(value, "name", str(value))]


class TraceReviewer:
    """Small helper for manually vetting captured traces."""

    def __init__(self, repository: TrainingRepository, *, reviewer: Optional[str] = None) -> None:
        self.repository = repository
        self.reviewer = reviewer

    def get_review_packet(self, record_id: str) -> dict[str, Any]:
        record = self.repository.load_raw(record_id)
        review = self.repository.load_review(record_id)

        top_evidence = []
        if record.decision is not None:
            top_evidence = [
                {
                    "category": chunk.artifacts.get("category", "unknown"),
                    "text": chunk.text,
                    "rank_score": chunk.rank_score,
                }
                for chunk in record.decision.top_evidence
            ]

        summary = {
            "record_id": record.record_id,
            "paper_id": record.paper_id,
            "classifier_kind": record.classifier_kind,
            "predicted_labels": _label_names(
                record.decision.result.classification if record.decision is not None else []
            ),
            "gold_label": _label_names(record.gold_label),
            "raw_completion": record.trace.raw_llm_response if record.trace is not None else None,
            "reasoning": (
                record.decision.result.evidence.get("reasoning")
                if record.decision is not None
                else None
            ),
            "top_evidence": top_evidence,
        }
        return {
            "record": record,
            "existing_review": review,
            "summary": summary,
        }

    def approve(
        self,
        record_id: str,
        *,
        buckets: list[TaskBucket],
        notes: str = "",
        edited_completion: Optional[str] = None,
        updated_label: Optional[list[str]] = None,
    ) -> ReviewDecision:
        review = ReviewDecision(
            record_id=record_id,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            reviewer=self.reviewer,
            status=ReviewStatus.APPROVED,
            buckets=list(buckets),
            notes=notes,
            edited_completion=edited_completion,
            updated_label=list(updated_label) if updated_label else None,
        )
        self.repository.save_review(review)
        return review

    def reject(
        self,
        record_id: str,
        *,
        notes: str = "",
    ) -> ReviewDecision:
        review = ReviewDecision(
            record_id=record_id,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            reviewer=self.reviewer,
            status=ReviewStatus.REJECTED,
            notes=notes,
        )
        self.repository.save_review(review)
        return review

    def mark_pending(
        self,
        record_id: str,
        *,
        notes: str = "",
    ) -> ReviewDecision:
        review = ReviewDecision(
            record_id=record_id,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            reviewer=self.reviewer,
            status=ReviewStatus.PENDING,
            notes=notes,
        )
        self.repository.save_review(review)
        return review
