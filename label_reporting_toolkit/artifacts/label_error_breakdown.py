#!/usr/bin/env python3
"""

python label_error_breakdown.py \
  --gt sampled_papers_full.csv \
  --output_dir output \
  --task geo \
  --model gemini-2-5-pro \
  --temperature 0.0 \
  --out_prefix label_analysis


python label_error_breakdown.py \
  --gt ../sampled_papers_full.csv \
  --output_dir ../output \
  --task geo \
  --model gemini-2-5-pro \
  --temperature 0.0 \
  --out_prefix label_analysis


Label distribution + per-label error analysis for the evaluation set.

This script is designed to complement the aggregate (micro/macro) metrics by:
  1) reporting label prevalence (support) in the evaluation set
  2) reporting per-label precision/recall/F1 (with mean ± SD across repeated runs)
  3) summarizing "error mass" (which labels contribute most FP / FN)
  4) (single-label task) reporting confusion matrices

Assumptions (adapt to your repo layout if needed):
  - Ground-truth file: sampled_papers_full.csv (TSV, with column 'paper_id')
  - Prediction files: output/<task>/<model>/temperature_<T>/**/final_*.tsv
    with columns: 'paper_id' and 'classification'

Notes:
  - For multi-label tasks, labels may be stored as comma-separated strings.
  - The canonicalization below mirrors your robustness choices (NFKC, strip, upper).
"""

from __future__ import annotations

import argparse
import ast
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, FrozenSet, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support


# ---------------------------------------------------------------------
# Canonicalization helpers (robust to brackets / curly quotes / spacing)
# ---------------------------------------------------------------------

_CURLY_TO_STRAIGHT = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
})

def _canon_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_CURLY_TO_STRAIGHT)
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def to_label_set(x) -> FrozenSet[str]:
    """Parse one raw cell into a frozenset of canonical labels."""
    if x is None:
        return frozenset()
    if isinstance(x, float) and pd.isna(x):
        return frozenset()
    if pd.isna(x):
        return frozenset()

    if isinstance(x, (list, tuple, set)):
        items = [str(i) for i in x if str(i).strip() != ""]
        return frozenset(_canon_text(i) for i in items if _canon_text(i))

    s = _canon_text(str(x))
    if not s:
        return frozenset()

    # Try to parse literals like "['A','B']"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                items = [str(i) for i in parsed if str(i).strip() != ""]
                return frozenset(_canon_text(i) for i in items if _canon_text(i))
            if isinstance(parsed, str):
                ps = _canon_text(parsed)
                return frozenset([ps]) if ps else frozenset()
        except Exception:
            pass

    # If literal_eval failed, strip outer brackets, then split on commas if present
    s_clean = re.sub(r"^\[|\]$", "", s).strip()
    if "," in s_clean:
        parts = [_canon_text(p) for p in s_clean.split(",")]
        parts = [p for p in parts if p]
        return frozenset(parts)

    return frozenset([_canon_text(s_clean)])


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

TASK_TO_GT_COL = {
    "geo":                "geo_classification",
    "data-type":          "data_type_classification",
    "data-accessibility": "availability_classification",
    "paper-type":         "ptype_classification",
}

@dataclass(frozen=True)
class Config:
    task: str
    model: str
    temperature: str

def find_result_files(output_dir: Path, cfg: Config) -> List[Path]:
    """Find all final_*.tsv files for a given task/model/temperature."""
    pattern = output_dir / cfg.task / cfg.model / f"temperature_{cfg.temperature}"
    return sorted(pattern.rglob("final_*.tsv"))

def load_gt(gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, sep="\t", dtype={"paper_id": str})
    gt["paper_id"] = gt["paper_id"].astype(str).str.strip()
    return gt

def load_preds_for_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"paper_id": str})
    if not {"paper_id", "classification"}.issubset(df.columns):
        raise ValueError(f"{path} missing required columns. Found: {list(df.columns)}")
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["pred_set"] = df["classification"].map(to_label_set)
    return df[["paper_id", "pred_set"]]

def align_gt_pred(gt: pd.DataFrame, cfg: Config, pred_df: pd.DataFrame) -> Tuple[List[FrozenSet[str]], List[FrozenSet[str]]]:
    gt_col = TASK_TO_GT_COL[cfg.task]
    if gt_col not in gt.columns:
        raise KeyError(f"GT missing column '{gt_col}'. Available: {list(gt.columns)}")

    gt_task = gt[["paper_id", gt_col]].copy()
    gt_task["gt_set"] = gt_task[gt_col].map(to_label_set)
    gt_task = gt_task.drop(columns=[gt_col])

    merged = pred_df.merge(gt_task, on="paper_id", how="inner")
    # Drop empty GT labels (if any)
    merged = merged[merged["gt_set"].map(len) > 0].copy()

    y_true = merged["gt_set"].tolist()
    y_pred = merged["pred_set"].tolist()
    return y_true, y_pred


# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------

def label_support(y_true: List[FrozenSet[str]]) -> pd.DataFrame:
    """Support = number of papers whose GT set contains the label."""
    counts: Dict[str, int] = {}
    for s in y_true:
        for lab in s:
            counts[lab] = counts.get(lab, 0) + 1
    out = pd.DataFrame({"label": list(counts.keys()), "support": list(counts.values())})
    return out.sort_values(["support", "label"], ascending=[False, True]).reset_index(drop=True)

def multilabel_binarize(y_true: List[FrozenSet[str]], y_pred: List[FrozenSet[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(set().union(*y_true, *y_pred))
    mlb = MultiLabelBinarizer(classes=labels)
    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)
    return Y_true, Y_pred, labels

def per_label_metrics_one_run(y_true: List[FrozenSet[str]], y_pred: List[FrozenSet[str]]) -> pd.DataFrame:
    """Per-label metrics for one run (binary relevance view)."""
    Y_true, Y_pred, labels = multilabel_binarize(y_true, y_pred)

    # Per-label PRF + support (support is count of true positives + false negatives)
    p, r, f1, sup = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0
    )

    tp = np.sum((Y_true == 1) & (Y_pred == 1), axis=0)
    fp = np.sum((Y_true == 0) & (Y_pred == 1), axis=0)
    fn = np.sum((Y_true == 1) & (Y_pred == 0), axis=0)
    pred_pos = np.sum(Y_pred == 1, axis=0)

    df = pd.DataFrame({
        "label": labels,
        "support": sup.astype(int),
        "pred_pos": pred_pos.astype(int),
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "precision": p,
        "recall": r,
        "f1": f1,
    })
    return df.sort_values(["support", "label"], ascending=[False, True]).reset_index(drop=True)

def aggregate_per_label_metrics(per_run: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate per-label metrics across runs: mean ± SD for precision/recall/F1,
    and mean counts for tp/fp/fn/pred_pos (counts can vary across runs).
    """
    if not per_run:
        raise ValueError("No per-run dataframes provided")

    all_labels = sorted(set().union(*[set(df["label"]) for df in per_run]))
    frames = []
    for i, df in enumerate(per_run):
        df_i = df.set_index("label").reindex(all_labels)
        df_i["run"] = i
        frames.append(df_i.reset_index())

    long = pd.concat(frames, ignore_index=True)

    # Support comes from GT; it should be identical across runs (but keep first non-null)
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return int(s2.iloc[0]) if len(s2) else 0

    agg = (
        long.groupby("label", as_index=False)
            .agg(
                support=("support", first_nonnull),
                pred_pos_mean=("pred_pos", "mean"),
                tp_mean=("tp", "mean"),
                fp_mean=("fp", "mean"),
                fn_mean=("fn", "mean"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
            )
    )
    # Replace NaN SD (single run) with 0
    for c in ["precision_std", "recall_std", "f1_std"]:
        agg[c] = agg[c].fillna(0.0)

    # Convenience: total error mass (mean FP+FN)
    agg["fp_fn_mean"] = agg["fp_mean"] + agg["fn_mean"]

    return agg.sort_values(["support", "label"], ascending=[False, True]).reset_index(drop=True)

def top_error_labels(agg: pd.DataFrame, k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-k labels by mean FP and by mean FN."""
    top_fp = agg.sort_values(["fp_mean", "support", "label"], ascending=[False, False, True]).head(k).reset_index(drop=True)
    top_fn = agg.sort_values(["fn_mean", "support", "label"], ascending=[False, False, True]).head(k).reset_index(drop=True)
    return top_fp, top_fn

def single_label_confusion(y_true: List[FrozenSet[str]], y_pred: List[FrozenSet[str]]) -> pd.DataFrame:
    """
    Confusion matrix for "single-label" tasks.

    If a prediction contains:
      - 0 labels: mapped to '__EMPTY__'
      - >1 labels: mapped to '__MULTI__'
      - 1 label: that label
    Same mapping for GT (though GT is expected to have exactly 1 label).
    """
    def as_single(s: FrozenSet[str]) -> str:
        if len(s) == 0:
            return "__EMPTY__"
        if len(s) > 1:
            return "__MULTI__"
        return next(iter(s))

    gt_single = [as_single(s) for s in y_true]
    pr_single = [as_single(s) for s in y_pred]

    cm = pd.crosstab(
        pd.Series(gt_single, name="gt"),
        pd.Series(pr_single, name="pred"),
        dropna=False,
    )
    return cm


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path("sampled_papers_full.csv"))
    ap.add_argument("--output_dir", type=Path, default=Path("output"))
    ap.add_argument("--task", type=str, required=True, choices=sorted(TASK_TO_GT_COL.keys()))
    ap.add_argument("--model", type=str, required=True, help="e.g., gemini-2-5-pro or gemini-2-5-flash")
    ap.add_argument("--temperature", type=str, required=True, help="e.g., 0.0 or 1.0")
    ap.add_argument("--out_prefix", type=str, default="label_analysis")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    cfg = Config(task=args.task, model=args.model, temperature=args.temperature)

    gt = load_gt(args.gt)

    files = find_result_files(args.output_dir, cfg)
    if not files:
        raise SystemExit(f"[ERROR] No result files found for: {cfg}")

    print(f"[INFO] Found {len(files)} runs for {cfg.task} / {cfg.model} / T={cfg.temperature}")

    per_run_metrics: List[pd.DataFrame] = []
    y_true_ref: Optional[List[FrozenSet[str]]] = None  # for support output

    for f in files:
        pred_df = load_preds_for_file(f)
        y_true, y_pred = align_gt_pred(gt, cfg, pred_df)

        if y_true_ref is None:
            y_true_ref = y_true

        per_run_metrics.append(per_label_metrics_one_run(y_true, y_pred))

    assert y_true_ref is not None

    # 1) Label support
    supp = label_support(y_true_ref)
    supp_path = Path(f"{args.out_prefix}_support_{cfg.task}.csv")
    supp.to_csv(supp_path, index=False)
    print(f"[OK] Wrote label support: {supp_path}")

    # 2) Per-label PRF aggregated across runs
    agg = aggregate_per_label_metrics(per_run_metrics)
    agg_path = Path(f"{args.out_prefix}_per_label_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[OK] Wrote per-label metrics: {agg_path}")

    # 3) Error concentration
    top_fp, top_fn = top_error_labels(agg, k=args.top_k)
    top_fp_path = Path(f"{args.out_prefix}_topFP_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv")
    top_fn_path = Path(f"{args.out_prefix}_topFN_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv")
    top_fp.to_csv(top_fp_path, index=False)
    top_fn.to_csv(top_fn_path, index=False)
    print(f"[OK] Wrote top-{args.top_k} FP labels: {top_fp_path}")
    print(f"[OK] Wrote top-{args.top_k} FN labels: {top_fn_path}")

    # 4) Confusion matrix for single-label task
    if cfg.task == "paper-type":
        # Use the first run only for a minimal diagnostic, or merge across runs if you prefer.
        # Here: merge across runs by concatenating (gt,pred) pairs across runs and summing counts.
        cm_total = None
        for df_run, f in zip(per_run_metrics, files):
            # Reload aligned pairs for the confusion matrix (needs per-sample labels)
            pred_df = load_preds_for_file(f)
            y_true, y_pred = align_gt_pred(gt, cfg, pred_df)
            cm = single_label_confusion(y_true, y_pred)
            cm_total = cm if cm_total is None else (cm_total.add(cm, fill_value=0))
        cm_total = cm_total.fillna(0).astype(int)
        cm_path = Path(f"{args.out_prefix}_confusion_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv")
        cm_total.to_csv(cm_path)
        print(f"[OK] Wrote confusion matrix: {cm_path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
