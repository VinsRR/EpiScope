from __future__ import annotations

TASK_TO_GT_COLUMN = {
    "paper_type": "ptype_classification",
    "paper-type": "ptype_classification",
    "data_accessibility": "availability_classification",
    "data-accessibility": "availability_classification",
    "data_type": "data_type_classification",
    "data-type": "data_type_classification",
    "geo": "geo_classification",
}

CANONICAL_TASK_NAMES = {
    "paper_type": "paper-type",
    "paper-type": "paper-type",
    "data_accessibility": "data-accessibility",
    "data-accessibility": "data-accessibility",
    "data_type": "data-type",
    "data-type": "data-type",
    "geo": "geo",
}


def canonical_task_name(task: str) -> str:
    canonical = CANONICAL_TASK_NAMES.get(task)
    if canonical is None:
        raise KeyError(f"Unsupported task name: {task!r}")
    return canonical


def ground_truth_column(task: str) -> str:
    column = TASK_TO_GT_COLUMN.get(task)
    if column is None:
        raise KeyError(f"No ground-truth column configured for task: {task!r}")
    return column
