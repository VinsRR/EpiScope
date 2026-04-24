from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from eval.common.io import read_ground_truth, read_results_tsv
from eval.common.labels import label_set_key, parse_label_set
from eval.common.paths import discover_result_files, parse_result_path
from eval.common.tasks import TASK_TO_GT_COLUMN, ground_truth_column


def load_merged_predictions(
    *,
    ground_truth_path: str | Path,
    run_roots: Iterable[str | Path],
    pattern: str = "final_*.tsv",
) -> pd.DataFrame:
    gt = read_ground_truth(ground_truth_path)

    gt_long = gt[["paper_id", *sorted(set(TASK_TO_GT_COLUMN.values()))]].melt(
        id_vars="paper_id",
        var_name="gt_col",
        value_name="gt_raw",
    )
    inverse = {column: task for task, column in TASK_TO_GT_COLUMN.items() if "-" in task}
    gt_long["task"] = gt_long["gt_col"].map(inverse)
    gt_long["gt_set"] = gt_long["gt_raw"].map(parse_label_set)
    gt_long = gt_long.drop(columns=["gt_col", "gt_raw"])
    gt_long = gt_long[gt_long["gt_set"].map(len) > 0].copy()
    gt_long = (
        gt_long.groupby(["paper_id", "task"], as_index=False)
        .agg(gt_set=("gt_set", lambda sets: frozenset().union(*sets)))
    )

    frames: list[pd.DataFrame] = []
    for result_path in discover_result_files(run_roots, pattern=pattern):
        try:
            df = read_results_tsv(result_path)
            meta = parse_result_path(result_path)
        except Exception:
            continue

        df["task"] = meta["task"]
        df["model"] = meta["model"]
        df["temperature"] = meta["temperature"]
        df["source_file"] = meta["source_file"]
        df["pred_set"] = df["classification"].map(parse_label_set)
        frames.append(df[["paper_id", "task", "model", "temperature", "source_file", "pred_set"]])

    if not frames:
        return pd.DataFrame(
            columns=["paper_id", "task", "model", "temperature", "source_file", "pred_set", "gt_set", "is_incorrect"]
        )

    predictions = pd.concat(frames, ignore_index=True)
    merged = predictions.merge(gt_long, on=["paper_id", "task"], how="inner")
    merged["is_incorrect"] = merged["pred_set"] != merged["gt_set"]
    return merged


def within_config_consistency(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["pred_key"] = df["pred_set"].map(label_set_key)

    per_paper = (
        df.groupby(["task", "model", "temperature", "paper_id"], as_index=False)
        .agg(
            n_runs=("pred_key", "count"),
            n_unique_preds=("pred_key", "nunique"),
            gt_set=("gt_set", "first"),
            always_wrong=("is_incorrect", "min"),
            ever_wrong=("is_incorrect", "max"),
        )
    )
    per_paper["is_stable"] = per_paper["n_unique_preds"] == 1

    summary = (
        per_paper.groupby(["task", "model", "temperature"], as_index=False)
        .agg(
            n_papers=("paper_id", "count"),
            n_stable=("is_stable", "sum"),
            n_unstable=("is_stable", lambda value: (~value).sum()),
            n_stable_wrong=("always_wrong", lambda value: ((per_paper.loc[value.index, "is_stable"]) & value).sum()),
            n_stable_correct=("always_wrong", lambda value: ((per_paper.loc[value.index, "is_stable"]) & (~value)).sum()),
        )
    )
    summary["pct_stable"] = (100 * summary["n_stable"] / summary["n_papers"]).round(1)
    return summary.sort_values(["task", "model", "temperature"]).reset_index(drop=True)


def unstable_papers(merged: pd.DataFrame, *, task: str, model: str, temperature: str) -> pd.DataFrame:
    df = merged[
        (merged["task"] == task)
        & (merged["model"] == model)
        & (merged["temperature"] == temperature)
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=["paper_id", "n_runs", "n_unique_preds", "preds", "gt_label"])

    df["pred_key"] = df["pred_set"].map(label_set_key)
    df["gt_key"] = df["gt_set"].map(label_set_key)

    per_paper = (
        df.groupby("paper_id", as_index=False)
        .agg(
            gt_key=("gt_key", "first"),
            n_runs=("pred_key", "count"),
            n_unique_preds=("pred_key", "nunique"),
            preds=("pred_key", lambda value: ", ".join(sorted(value.unique()))),
        )
    )
    out = per_paper[per_paper["n_unique_preds"] > 1].copy()
    out["gt_label"] = out["gt_key"].map(lambda key: key.split("|") if key else [])
    return out.drop(columns=["gt_key"]).sort_values("paper_id").reset_index(drop=True)


def containment_pro_vs_flash(
    merged: pd.DataFrame,
    *,
    temperature: str = "0.0",
    pro_model: str = "gemini-2-5-pro",
    flash_model: str = "gemini-2-5-flash",
) -> pd.DataFrame:
    subset = merged[merged["temperature"] == temperature].copy()

    rows = []
    for task in sorted({task for task in TASK_TO_GT_COLUMN if "-" in task}):
        task_df = subset[subset["task"] == task]

        def ever_wrong_set(model_name: str) -> set[str]:
            model_df = task_df[task_df["model"] == model_name]
            if model_df.empty:
                return set()
            ever_wrong = model_df.groupby("paper_id")["is_incorrect"].max()
            return set(ever_wrong[ever_wrong].index)

        pro_errors = ever_wrong_set(pro_model)
        flash_errors = ever_wrong_set(flash_model)
        shared = pro_errors & flash_errors
        union = pro_errors | flash_errors

        rows.append(
            {
                "task": task,
                "temperature": temperature,
                "n_pro_errors": len(pro_errors),
                "n_flash_errors": len(flash_errors),
                "n_shared": len(shared),
                "pro_in_flash_pct": round(100 * len(shared) / len(pro_errors), 1) if pro_errors else None,
                "flash_in_pro_pct": round(100 * len(shared) / len(flash_errors), 1) if flash_errors else None,
                "jaccard_pct": round(100 * len(shared) / len(union), 1) if union else None,
                "n_pro_only": len(pro_errors - flash_errors),
                "n_flash_only": len(flash_errors - pro_errors),
            }
        )

    return pd.DataFrame(rows)


def pro_always_wrong(
    merged: pd.DataFrame,
    *,
    temperature: str = "0.0",
    pro_model: str = "gemini-2-5-pro",
) -> pd.DataFrame:
    subset = merged[(merged["model"] == pro_model) & (merged["temperature"] == temperature)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["task", "paper_id", "gt_set", "n_runs", "wrong_runs", "top_wrong_pred"])

    per_paper = (
        subset.groupby(["task", "paper_id"], as_index=False)
        .agg(
            gt_set=("gt_set", "first"),
            n_runs=("is_incorrect", "count"),
            wrong_runs=("is_incorrect", "sum"),
        )
    )
    always_wrong = per_paper[per_paper["wrong_runs"] == per_paper["n_runs"]].copy()
    if always_wrong.empty:
        always_wrong["top_wrong_pred"] = pd.Series(dtype=object)
        return always_wrong

    wrong_only = subset[subset["is_incorrect"]].copy()
    wrong_only["pred_key"] = wrong_only["pred_set"].map(label_set_key)
    top_wrong = (
        wrong_only.groupby(["task", "paper_id"])["pred_key"]
        .agg(lambda value: value.mode().iloc[0] if not value.mode().empty else "")
        .rename("top_wrong_key")
        .reset_index()
    )
    top_wrong["top_wrong_pred"] = top_wrong["top_wrong_key"].map(
        lambda key: frozenset(key.split("|")) if key else frozenset()
    )
    top_wrong = top_wrong.drop(columns=["top_wrong_key"])

    return (
        always_wrong.merge(top_wrong, on=["task", "paper_id"], how="left")
        .sort_values(["task", "paper_id"])
        .reset_index(drop=True)
    )


def per_run_metrics(
    *,
    ground_truth_path: str | Path,
    run_roots: Iterable[str | Path],
    pattern: str = "final_*.tsv",
) -> pd.DataFrame:
    from eval.classification.metrics import (
        jaccard_samples,
        multilabel_hamming_loss,
        multilabel_prf,
        subset_accuracy,
    )

    gt = read_ground_truth(ground_truth_path)
    rows: list[dict] = []

    for result_path in discover_result_files(run_roots, pattern=pattern):
        try:
            df = read_results_tsv(result_path)
            meta = parse_result_path(result_path)
        except Exception:
            continue

        gt_col = ground_truth_column(meta["task"])
        if gt_col not in gt.columns:
            raise KeyError(f"Ground-truth file is missing expected column {gt_col!r}")

        gt_dict = gt.set_index("paper_id")[gt_col].astype(str).to_dict()
        pred_dict = df.set_index("paper_id")["classification"].astype(str).to_dict()

        prf_micro = multilabel_prf(gt_dict, pred_dict, average="micro", sep=",")
        prf_macro = multilabel_prf(gt_dict, pred_dict, average="macro", sep=",")
        rows.append(
            {
                "task": meta["task"],
                "model": meta["model"],
                "temperature": meta["temperature"],
                "source_file": meta["source_file"],
                "jaccard_samples": jaccard_samples(gt_dict, pred_dict, sep=","),
                "micro_precision": prf_micro["precision"],
                "micro_recall": prf_micro["recall"],
                "micro_f1": prf_micro["f1"],
                "macro_precision": prf_macro["precision"],
                "macro_recall": prf_macro["recall"],
                "macro_f1": prf_macro["f1"],
                "hamming_loss": multilabel_hamming_loss(gt_dict, pred_dict, sep=","),
                "subset_accuracy": subset_accuracy(gt_dict, pred_dict, sep=","),
                "n_predictions": len(pred_dict),
            }
        )

    return pd.DataFrame(rows)
