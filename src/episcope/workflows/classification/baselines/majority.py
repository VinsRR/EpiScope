from __future__ import annotations

from collections.abc import Mapping, Sequence

from episcope.schemas import PaperMetadata
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    labels_from_names,
    majority_label_set,
    parse_label_names,
    result_from_labels,
)


class MajorityLabelBaseline:
    """Predict the most frequent gold label set.

    This is intentionally simple. In leave-one-out mode the runner can provide a
    paper-specific majority label set computed without that paper's own gold
    label, avoiding direct leakage while preserving the class-imbalance baseline.
    """

    name = "majority"

    def __init__(
        self,
        *,
        classifier_kind: str,
        label_names: Sequence[str],
        per_paper_label_names: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.label_names = tuple(label_names)
        self.per_paper_label_names = {
            str(key): tuple(value)
            for key, value in (per_paper_label_names or {}).items()
        }

    @classmethod
    def from_records(
        cls,
        *,
        classifier_kind: str,
        records: Sequence[Mapping[str, object]],
        ground_truth_column: str,
        leave_one_out: bool = True,
    ) -> "MajorityLabelBaseline":
        rows = [
            (
                str(record.get("paper_id")),
                parse_label_names(record.get(ground_truth_column)),
            )
            for record in records
        ]
        global_majority = majority_label_set(label_set for _, label_set in rows)
        per_paper: dict[str, Sequence[str]] = {}
        if leave_one_out:
            for paper_id, _ in rows:
                per_paper[paper_id] = (
                    majority_label_set(
                        label_set
                        for other_id, label_set in rows
                        if other_id != paper_id
                    )
                    or global_majority
                )
        return cls(
            classifier_kind=classifier_kind,
            label_names=global_majority,
            per_paper_label_names=per_paper,
        )

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        label_names = self.per_paper_label_names.get(str(paper_id), self.label_names)
        labels = labels_from_names(self.classifier_kind, label_names)
        result = result_from_labels(
            self.classifier_kind,
            labels,
            confidence=1.0,
            reasoning=(
                "Majority-label baseline: predicted the most frequent gold label "
                "set for this task."
            ),
            class_probabilities={name: 1.0 for name in label_names},
        )
        return BaselinePrediction(result=result)
