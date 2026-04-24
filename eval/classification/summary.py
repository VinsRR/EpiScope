from __future__ import annotations

import pandas as pd

from .metrics import summarize_metric_values

METRIC_COLUMNS = [
    "jaccard_samples",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "hamming_loss",
    "subset_accuracy",
]


def summarize_per_run_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    grouped = per_run.groupby(["task", "model", "temperature"], as_index=False)
    for (task, model, temperature), frame in grouped:
        row = {
            "task": task,
            "model": model,
            "temperature": temperature,
            "n_files": len(frame),
        }
        for column in METRIC_COLUMNS:
            stats = summarize_metric_values(frame[column].tolist())
            for stat_name, stat_value in stats.items():
                row[f"{column}_{stat_name}"] = stat_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["task", "model", "temperature"]).reset_index(drop=True)
