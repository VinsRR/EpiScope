from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .schemas import (
    CapturedTraceRecord,
    ReviewDecision,
    ReviewStatus,
    TaskBucket,
    deserialize_captured_trace_record,
    deserialize_review_decision,
    serialize_captured_trace_record,
    serialize_review_decision,
)


class TrainingRepository:
    """Filesystem-backed storage for raw captured traces and review decisions."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw"

    @property
    def reviews_dir(self) -> Path:
        return self.root_dir / "reviews"

    @property
    def exports_dir(self) -> Path:
        return self.root_dir / "exports"

    def _write_json(self, path: Path, data: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def save_raw(self, record: CapturedTraceRecord) -> Path:
        return self._write_json(
            self.raw_dir / f"{record.record_id}.json",
            serialize_captured_trace_record(record),
        )

    def load_raw(self, record_id: str) -> CapturedTraceRecord:
        path = self.raw_dir / f"{record_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_captured_trace_record(data)

    def iter_raw(self) -> Iterable[CapturedTraceRecord]:
        for path in sorted(self.raw_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            yield deserialize_captured_trace_record(data)

    def save_review(self, review: ReviewDecision) -> Path:
        return self._write_json(
            self.reviews_dir / f"{review.record_id}.json",
            serialize_review_decision(review),
        )

    def load_review(self, record_id: str) -> Optional[ReviewDecision]:
        path = self.reviews_dir / f"{record_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_review_decision(data)

    def iter_reviews(self) -> Iterable[ReviewDecision]:
        for path in sorted(self.reviews_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            yield deserialize_review_decision(data)

    def iter_pending(self) -> Iterable[CapturedTraceRecord]:
        for record in self.iter_raw():
            review = self.load_review(record.record_id)
            if review is None or review.status == ReviewStatus.PENDING:
                yield record

    def iter_approved(
        self,
        *,
        bucket: Optional[TaskBucket] = None,
    ) -> Iterable[tuple[CapturedTraceRecord, ReviewDecision]]:
        for record in self.iter_raw():
            review = self.load_review(record.record_id)
            if review is None or review.status != ReviewStatus.APPROVED:
                continue
            if bucket is not None and bucket not in review.buckets:
                continue
            yield record, review
