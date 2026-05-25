"""Per-label metric computation for multilabel classification results."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from eval.common.io import read_ground_truth, read_results_tsv
from eval.common.paths import discover_result_files, parse_result_path
from eval.common.tasks import CANONICAL_TASK_NAMES, TASK_TO_GT_COLUMN


# ---------------------------------------------------------------------------
# Internal helpers (replicate _normalize_label_set / _binarize logic
# without importing from metrics, so we can control the label universe).
# ---------------------------------------------------------------------------

def _normalize_label_set(value, sep: str = ",") -> set[str]:
    """Normalize one entry into a set of non-empty, stripped strings.

    Handles Python list notation (``"['A', 'B']"``) produced by the pipeline,
    plain comma-separated strings, and actual list/set/tuple objects.
    """
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    # Try to parse Python list notation first (e.g. "['EUROPE', 'NORTH_AMERICA']").
    # Some GT entries contain Unicode curly quotes (U+2018/U+2019) instead of
    # straight ASCII apostrophes; normalise before calling literal_eval.
    if text.startswith("[") or text.startswith("‘") or text.startswith("“"):
        normalised = text.replace("‘", "'").replace("’", "'") \
                         .replace("“", '"').replace("”", '"')
        try:
            parsed = ast.literal_eval(normalised)
            if isinstance(parsed, list):
                return {str(item).strip() for item in parsed if str(item).strip()}
        except (ValueError, SyntaxError):
            pass
    return {part.strip() for part in text.split(sep) if part.strip()}


def _binarize_with_classes(
    y_true_raw,
    y_pred_raw,
    classes: list[str],
    sep: str = ",",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Binarize y_true / y_pred using a *fixed* label vocabulary.

    Parameters
    ----------
    y_true_raw, y_pred_raw : sequence of raw label strings (one per sample)
    classes : fixed vocabulary (defines column order in the output matrices)
    sep : separator used to split comma-joined label strings

    Returns
    -------
    y_true_bin, y_pred_bin : (n_samples, n_classes) indicator arrays
    classes_out : the ordered label list (same as ``classes``)
    """
    y_true_sets = [_normalize_label_set(v, sep) for v in y_true_raw]
    y_pred_sets = [_normalize_label_set(v, sep) for v in y_pred_raw]

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])  # fit to establish the fixed class order
    y_true_bin = mlb.transform(y_true_sets)
    y_pred_bin = mlb.transform(y_pred_sets)
    return y_true_bin, y_pred_bin, list(mlb.classes_)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def per_label_metrics_for_files(
    *,
    ground_truth_path: str | Path,
    run_roots: Iterable[str | Path],
    pattern: str = "final_*.tsv",
) -> pd.DataFrame:
    """
    Compute per-label precision / recall / F1 for every result file found
    under *run_roots*.

    Returns a DataFrame with columns:
      task, model, temperature, source_file, label, support,
      precision, recall, f1

    One row per (source_file, label).  Labels with zero support in the
    ground-truth are excluded.  The ground-truth label vocabulary is
    determined per task from the *ground_truth_path* file.
    """
    gt = read_ground_truth(ground_truth_path)

    # Build per-task ground-truth dictionaries and label vocabularies.
    # We derive them directly from the GT file so the vocabulary is stable.
    task_gt: dict[str, dict[str, str]] = {}
    task_classes: dict[str, list[str]] = {}
    task_support: dict[str, dict[str, int]] = {}

    # Deduplicate TASK_TO_GT_COLUMN by canonical task name so we process each
    # task exactly once.  This also ensures "geo" (no hyphen) is included.
    canonical_to_gt_col: dict[str, str] = {}
    for k, gt_col in TASK_TO_GT_COLUMN.items():
        canonical = CANONICAL_TASK_NAMES.get(k, k)
        if canonical not in canonical_to_gt_col:
            canonical_to_gt_col[canonical] = gt_col

    for task, gt_col in canonical_to_gt_col.items():
        if gt_col not in gt.columns:
            continue
        gt_col_series = gt.set_index("paper_id")[gt_col].astype(str)
        task_gt[task] = gt_col_series.to_dict()

        # Build vocabulary from GT — values use Python list notation.
        label_counts: dict[str, int] = {}
        for raw in gt_col_series:
            for lbl in _normalize_label_set(raw):
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Use support > 0 labels only, ordered alphabetically for stability
        classes = sorted(lbl for lbl, cnt in label_counts.items() if cnt > 0)
        task_classes[task] = classes
        task_support[task] = label_counts

    rows: list[dict] = []

    for result_path in discover_result_files(run_roots, pattern=pattern):
        try:
            df = read_results_tsv(result_path)
            meta = parse_result_path(result_path)
        except Exception:
            continue

        task = meta["task"]
        if task not in task_gt:
            continue

        classes = task_classes[task]
        if not classes:
            continue

        gt_dict = task_gt[task]
        # Align predictions with GT
        pred_dict = df.set_index("paper_id")["classification"].astype(str).to_dict()

        # Intersect on paper_id
        common_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
        if not common_ids:
            continue

        # Both GT and predictions use Python list notation e.g. "['A', 'B']".
        # _normalize_label_set handles this via ast.literal_eval internally.
        y_true_raw = [gt_dict[pid] for pid in common_ids]
        y_pred_raw = [pred_dict[pid] for pid in common_ids]

        y_true_sets = [_normalize_label_set(v) for v in y_true_raw]
        y_pred_sets = [_normalize_label_set(v) for v in y_pred_raw]

        mlb = MultiLabelBinarizer(classes=classes)
        mlb.fit([classes])
        y_true_bin = mlb.transform(y_true_sets)
        y_pred_bin = mlb.transform(y_pred_sets)

        precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            average=None,
            labels=list(range(len(classes))),
            zero_division=0,
        )

        for i, label in enumerate(classes):
            gt_support = task_support[task].get(label, 0)
            if gt_support == 0:
                continue
            rows.append(
                {
                    "task": task,
                    "model": meta["model"],
                    "temperature": meta["temperature"],
                    "source_file": meta["source_file"],
                    "label": label,
                    "support": int(gt_support),
                    "precision": float(precision_arr[i]),
                    "recall": float(recall_arr[i]),
                    "f1": float(f1_arr[i]),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["task", "model", "temperature", "source_file", "label",
                 "support", "precision", "recall", "f1"],
    )


def summarize_per_label_metrics(per_label: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate *per_label* (one row per source_file × label) over source_file.

    Returns a DataFrame with columns:
      task, model, temperature, label, support,
      precision_mean, precision_std,
      recall_mean, recall_std,
      f1_mean, f1_std
    """
    if per_label.empty:
        return pd.DataFrame(
            columns=[
                "task", "model", "temperature", "label", "support",
                "precision_mean", "precision_std",
                "recall_mean", "recall_std",
                "f1_mean", "f1_std",
            ]
        )

    rows: list[dict] = []
    group_keys = ["task", "model", "temperature", "label"]
    for key, frame in per_label.groupby(group_keys, as_index=False):
        task, model, temperature, label = key
        values = frame[["precision", "recall", "f1"]].values  # (n_files, 3)
        n = len(frame)
        ddof = 1 if n > 1 else 0
        rows.append(
            {
                "task": task,
                "model": model,
                "temperature": temperature,
                "label": label,
                "support": int(frame["support"].iloc[0]),
                "precision_mean": float(np.mean(values[:, 0])),
                "precision_std": float(np.std(values[:, 0], ddof=ddof)),
                "recall_mean": float(np.mean(values[:, 1])),
                "recall_std": float(np.std(values[:, 1], ddof=ddof)),
                "f1_mean": float(np.mean(values[:, 2])),
                "f1_std": float(np.std(values[:, 2], ddof=ddof)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["task", "model", "temperature", "label"])
        .reset_index(drop=True)
    )
