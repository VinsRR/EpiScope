from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def _normalize_label_set(value: Any, sep: str = ";") -> set[str]:
    """Normalize one entry into a set of non-empty, stripped strings."""
    if value is None:
        return set()

    if isinstance(value, (set, frozenset, list, tuple)):
        out = set()
        for item in value:
            text = str(item).strip()
            if text:
                out.add(text)
        return out

    text = str(value).strip()
    if not text:
        return set()

    return {part.strip() for part in text.split(sep) if part.strip()}


def _align_inputs(y_true, y_pred):
    """
    Align inputs:
    - dicts: align on key intersection using deterministic key order
    - sequences: align by position
    """
    if isinstance(y_true, Mapping) and isinstance(y_pred, Mapping):
        common_keys = sorted(y_true.keys() & y_pred.keys())
        return [y_true[key] for key in common_keys], [y_pred[key] for key in common_keys]

    if len(y_true) != len(y_pred):
        raise ValueError("List/sequence inputs must have the same length")
    return y_true, y_pred


def _binarize(y_true, y_pred, sep: str = ";") -> Tuple[np.ndarray, np.ndarray]:
    """Convert aligned multilabel data into binary indicator matrices."""
    y_true, y_pred = _align_inputs(y_true, y_pred)

    y_true_sets = [_normalize_label_set(item, sep) for item in y_true]
    y_pred_sets = [_normalize_label_set(item, sep) for item in y_pred]

    labels = sorted(set().union(*y_true_sets, *y_pred_sets))
    if not labels:
        n_rows = len(y_true_sets)
        return np.zeros((n_rows, 0), dtype=int), np.zeros((n_rows, 0), dtype=int)

    mlb = MultiLabelBinarizer(classes=labels)
    y_true_bin = mlb.fit_transform(y_true_sets)
    y_pred_bin = mlb.transform(y_pred_sets)
    return y_true_bin, y_pred_bin


def jaccard_samples(y_true, y_pred, sep: str = ";") -> float:
    y_true_bin, y_pred_bin = _binarize(y_true, y_pred, sep)
    if y_true_bin.shape[1] == 0:
        return 1.0
    return float(jaccard_score(y_true_bin, y_pred_bin, average="samples", zero_division=0))


def multilabel_prf(y_true, y_pred, average: str = "micro", sep: str = ";") -> dict[str, float]:
    y_true_bin, y_pred_bin = _binarize(y_true, y_pred, sep)
    if y_true_bin.shape[1] == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    return {
        "precision": float(precision_score(y_true_bin, y_pred_bin, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, average=average, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, average=average, zero_division=0)),
    }


def multilabel_hamming_loss(y_true, y_pred, sep: str = ";") -> float:
    y_true_bin, y_pred_bin = _binarize(y_true, y_pred, sep)
    if y_true_bin.shape[1] == 0:
        return 0.0
    return float(hamming_loss(y_true_bin, y_pred_bin))


def subset_accuracy(y_true, y_pred, sep: str = ";") -> float:
    y_true_bin, y_pred_bin = _binarize(y_true, y_pred, sep)
    if y_true_bin.shape[1] == 0:
        return 1.0
    return float(accuracy_score(y_true_bin, y_pred_bin))


def summarize_metric_values(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mad": float(np.median(np.abs(array - np.median(array)))),
    }
