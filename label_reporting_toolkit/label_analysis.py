#!/usr/bin/env python3
"""
label_analysis.py  —  unified label analysis pipeline
======================================================

Replaces label_error_breakdown.py + label_artifacts_from_csv.py with a single
entry point that can run the full pipeline (compute → save CSVs → render
outputs) or re-render outputs from previously saved CSVs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK-START EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Full pipeline for ONE task (compute metrics + save CSVs + all plots + LaTeX):

    python label_analysis.py \
        --model gemini-2-5-pro --temperature 0.0 \
        --tasks geo \
        --emit csv plots latex

2. Full pipeline for ALL tasks at once:

    python label_analysis.py \
        --model gemini-2-5-pro --temperature 0.0 \
        --tasks geo data-type data-accessibility paper-type \
        --emit csv plots latex

3. Just the combined prevalence grid (all tasks, from raw predictions):

    python label_analysis.py \
        --model gemini-2-5-pro --temperature 0.0 \
        --tasks geo data-type data-accessibility paper-type \
        --emit plots --plot_types prevalence_grid

4. Re-render plots from already-saved CSVs (no predictions needed):

    python label_analysis.py \
        --from_csv \
        --tasks geo data-type data-accessibility paper-type \
        --model gemini-2-5-pro --temperature 0.0 \
        --emit plots --plot_types prevalence_grid

5. Only LaTeX tables, no plots:

    python label_analysis.py \
        --model gemini-2-5-pro --temperature 0.0 \
        --tasks geo data-type \
        --emit csv latex

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE LAYOUT (defaults, all overridable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Inputs
  ------
  Ground-truth TSV  : --gt             (default: ../sampled_papers_full.csv)
  Prediction files  : --pred_dir       (default: ../output/)
                      layout: <pred_dir>/<task>/<model>/temperature_<T>/**/final_*.tsv

  Outputs (all written under --out_dir, default: artifacts/)
  -----------------------------------------------------------
  CSVs
    per-label metrics : <prefix>_per_label_<task>_<model>_T<temp>.csv
    label support     : <prefix>_support_<task>.csv
    top-FP labels     : <prefix>_topFP_<task>_<model>_T<temp>.csv
    top-FN labels     : <prefix>_topFN_<task>_<model>_T<temp>.csv
    confusion matrix  : <prefix>_confusion_<task>_<model>_T<temp>.csv  (paper-type only)

  Plots  (in <out_dir>/plots/)
    per-label F1 bar      : <prefix>_f1bar_<task>_<model>_T<temp>.png
    support bar           : <prefix>_support_<task>.png
    top-FP / top-FN bars  : <prefix>_topFP/FN_<task>_<model>_T<temp>.png
    prevalence scatter    : <prefix>_prevalence_<task>_<model>_T<temp>.png
    prevalence grid       : <prefix>_prevalence_grid_<model>_T<temp>.png

  LaTeX  (in <out_dir>/latex/)
    per-task tables for support, PRF, top-FP, top-FN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOT TYPES  (--plot_types, default: all)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  f1bar            per-label F1 with error bars (one plot per task)
  support          label support bar chart (one plot per task)
  top_errors       top-FP and top-FN bar charts (one pair per task)
  prevalence       prevalence vs P/R/F1 scatter (one plot per task)
  prevalence_grid  combined grid of prevalence plots for all requested tasks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_TO_GT_COL: Dict[str, str] = {
    "geo":                "geo_classification",
    "data-type":          "data_type_classification",
    "data-accessibility": "availability_classification",
    "paper-type":         "ptype_classification",
}

TASK_DISPLAY: Dict[str, str] = {
    "geo":                "Geography",
    "data-type":          "Data Type",
    "data-accessibility": "Data Accessibility",
    "paper-type":         "Paper Type",
}

ALL_TASKS = list(TASK_TO_GT_COL.keys())
ALL_PLOT_TYPES = ["f1bar", "support", "top_errors", "prevalence", "prevalence_grid"]

_METRIC_STYLES: Dict[str, dict] = {
    "F1":        {"color": "#2196F3", "marker": "o", "zorder": 4},
    "Precision": {"color": "#E91E63", "marker": "s", "zorder": 3},
    "Recall":    {"color": "#4CAF50", "marker": "^", "zorder": 3},
}


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalization
# ─────────────────────────────────────────────────────────────────────────────

_CURLY_TO_STRAIGHT = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
})


def _canon_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_CURLY_TO_STRAIGHT)
    return re.sub(r"\s+", " ", s.strip().upper())


def to_label_set(x) -> FrozenSet[str]:
    """Parse one raw prediction/GT cell into a frozenset of canonical labels."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return frozenset()
    try:
        if pd.isna(x):
            return frozenset()
    except (TypeError, ValueError):
        pass

    if isinstance(x, (list, tuple, set)):
        return frozenset(c for i in x if (c := _canon_text(str(i))))

    s = _canon_text(str(x))
    if not s:
        return frozenset()

    if s[0] in "([{" and s[-1] in ")]}":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return frozenset(c for i in parsed if (c := _canon_text(str(i))))
            if isinstance(parsed, str) and (c := _canon_text(parsed)):
                return frozenset([c])
        except Exception:
            pass

    s_clean = re.sub(r"^\[|\]$", "", s).strip()
    if "," in s_clean:
        return frozenset(c for p in s_clean.split(",") if (c := _canon_text(p)))

    return frozenset([s_clean]) if s_clean else frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    task: str
    model: str
    temperature: str


def find_result_files(pred_dir: Path, cfg: Config) -> List[Path]:
    pattern = pred_dir / cfg.task / cfg.model / f"temperature_{cfg.temperature}"
    return sorted(pattern.rglob("final_*.tsv"))


def load_gt(gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, sep="\t", dtype={"paper_id": str})
    gt["paper_id"] = gt["paper_id"].astype(str).str.strip()
    return gt


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"paper_id": str})
    if not {"paper_id", "classification"}.issubset(df.columns):
        raise ValueError(f"{path}: missing required columns. Found: {list(df.columns)}")
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["pred_set"] = df["classification"].map(to_label_set)
    return df[["paper_id", "pred_set"]]


def align(gt: pd.DataFrame, cfg: Config,
          pred_df: pd.DataFrame) -> Tuple[List[FrozenSet[str]], List[FrozenSet[str]]]:
    gt_col = TASK_TO_GT_COL[cfg.task]
    if gt_col not in gt.columns:
        raise KeyError(f"GT missing column '{gt_col}'. Available: {list(gt.columns)}")
    gt_task = gt[["paper_id", gt_col]].copy()
    gt_task["gt_set"] = gt_task[gt_col].map(to_label_set)
    merged = (pred_df
              .merge(gt_task[["paper_id", "gt_set"]], on="paper_id", how="inner")
              .loc[lambda d: d["gt_set"].map(len) > 0])
    return merged["gt_set"].tolist(), merged["pred_set"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _binarize(y_true: List[FrozenSet[str]],
              y_pred: List[FrozenSet[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(set().union(*y_true, *y_pred))
    mlb = MultiLabelBinarizer(classes=labels)
    return mlb.fit_transform(y_true), mlb.transform(y_pred), labels


def per_label_metrics_one_run(y_true: List[FrozenSet[str]],
                               y_pred: List[FrozenSet[str]]) -> pd.DataFrame:
    Y_true, Y_pred, labels = _binarize(y_true, y_pred)
    p, r, f1, sup = precision_recall_fscore_support(Y_true, Y_pred,
                                                    average=None, zero_division=0)
    tp = np.sum((Y_true == 1) & (Y_pred == 1), axis=0)
    fp = np.sum((Y_true == 0) & (Y_pred == 1), axis=0)
    fn = np.sum((Y_true == 1) & (Y_pred == 0), axis=0)
    return pd.DataFrame({
        "label":     labels,
        "support":   sup.astype(int),
        "pred_pos":  np.sum(Y_pred == 1, axis=0).astype(int),
        "tp": tp.astype(int), "fp": fp.astype(int), "fn": fn.astype(int),
        "precision": p, "recall": r, "f1": f1,
    }).sort_values(["support", "label"], ascending=[False, True]).reset_index(drop=True)


def aggregate_runs(per_run: List[pd.DataFrame]) -> pd.DataFrame:
    all_labels = sorted(set().union(*[set(df["label"]) for df in per_run]))
    long = pd.concat([
        df.set_index("label").reindex(all_labels).assign(run=i).reset_index()
        for i, df in enumerate(per_run)
    ], ignore_index=True)

    def first_nonnull(s):
        s2 = s.dropna()
        return int(s2.iloc[0]) if len(s2) else 0

    agg = (long.groupby("label", as_index=False).agg(
        support      =("support",   first_nonnull),
        pred_pos_mean=("pred_pos",  "mean"),
        tp_mean      =("tp",        "mean"),  tp_std =("tp",  "std"),
        fp_mean      =("fp",        "mean"),  fp_std =("fp",  "std"),
        fn_mean      =("fn",        "mean"),  fn_std =("fn",  "std"),
        precision_mean=("precision","mean"), precision_std=("precision","std"),
        recall_mean   =("recall",   "mean"), recall_std   =("recall",  "std"),
        f1_mean       =("f1",       "mean"), f1_std       =("f1",      "std"),
    ))
    for c in ["precision_std", "recall_std", "f1_std", "fp_std", "fn_std", "tp_std"]:
        agg[c] = agg[c].fillna(0.0)
    agg["fp_fn_mean"] = agg["fp_mean"] + agg["fn_mean"]
    return agg.sort_values(["support", "label"], ascending=[False, True]).reset_index(drop=True)


def label_support(y_true: List[FrozenSet[str]]) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for s in y_true:
        for lab in s:
            counts[lab] = counts.get(lab, 0) + 1
    return (pd.DataFrame({"label": list(counts), "support": list(counts.values())})
            .sort_values(["support", "label"], ascending=[False, True])
            .reset_index(drop=True))


def top_error_labels(agg: pd.DataFrame,
                     k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_fp = agg.sort_values(["fp_mean", "support", "label"],
                             ascending=[False, False, True]).head(k).reset_index(drop=True)
    top_fn = agg.sort_values(["fn_mean", "support", "label"],
                             ascending=[False, False, True]).head(k).reset_index(drop=True)
    return top_fp, top_fn


def single_label_confusion(y_true: List[FrozenSet[str]],
                            y_pred: List[FrozenSet[str]]) -> pd.DataFrame:
    def _single(s):
        if len(s) == 0: return "__EMPTY__"
        if len(s) > 1:  return "__MULTI__"
        return next(iter(s))
    return pd.crosstab(
        pd.Series([_single(s) for s in y_true], name="gt"),
        pd.Series([_single(s) for s in y_pred], name="pred"),
        dropna=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV filename helpers
# ─────────────────────────────────────────────────────────────────────────────

def _csv_per_label(out_dir: Path, prefix: str, cfg: Config) -> Path:
    return out_dir / f"{prefix}_per_label_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv"

def _csv_support(out_dir: Path, prefix: str, cfg: Config) -> Path:
    return out_dir / f"{prefix}_support_{cfg.task}.csv"

def _csv_topfp(out_dir: Path, prefix: str, cfg: Config) -> Path:
    return out_dir / f"{prefix}_topFP_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv"

def _csv_topfn(out_dir: Path, prefix: str, cfg: Config) -> Path:
    return out_dir / f"{prefix}_topFN_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv"

def _csv_confusion(out_dir: Path, prefix: str, cfg: Config) -> Path:
    return out_dir / f"{prefix}_confusion_{cfg.task}_{cfg.model}_T{cfg.temperature}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Compute + save CSVs for one task
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_save(gt: pd.DataFrame, cfg: Config, pred_dir: Path,
                     out_dir: Path, prefix: str, top_k: int) -> pd.DataFrame:
    """
    Run the full metric computation for one task/model/temperature,
    write CSVs, and return the aggregated per-label DataFrame.
    """
    files = find_result_files(pred_dir, cfg)
    if not files:
        raise SystemExit(f"[ERROR] No result files found for: {cfg}\n"
                         f"  Expected under: {pred_dir / cfg.task / cfg.model}")

    print(f"[INFO] {cfg.task}: found {len(files)} run(s)")

    per_run: List[pd.DataFrame] = []
    y_true_ref = None
    gt_pred_pairs: List[Tuple] = []   # keep for confusion matrix

    for f in files:
        pred_df = load_preds(f)
        y_true, y_pred = align(gt, cfg, pred_df)
        if y_true_ref is None:
            y_true_ref = y_true
        per_run.append(per_label_metrics_one_run(y_true, y_pred))
        gt_pred_pairs.append((y_true, y_pred))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Support
    supp = label_support(y_true_ref)
    supp.to_csv(_csv_support(out_dir, prefix, cfg), index=False)

    # Per-label aggregated
    agg = aggregate_runs(per_run)
    agg.to_csv(_csv_per_label(out_dir, prefix, cfg), index=False)

    # Top errors
    top_fp, top_fn = top_error_labels(agg, k=top_k)
    top_fp.to_csv(_csv_topfp(out_dir, prefix, cfg), index=False)
    top_fn.to_csv(_csv_topfn(out_dir, prefix, cfg), index=False)

    # Confusion (paper-type only)
    if cfg.task == "paper-type":
        cm_total = None
        for y_true, y_pred in gt_pred_pairs:
            cm = single_label_confusion(y_true, y_pred)
            cm_total = cm if cm_total is None else cm_total.add(cm, fill_value=0)
        cm_total.fillna(0).astype(int).to_csv(_csv_confusion(out_dir, prefix, cfg))

    print(f"  → CSVs written to {out_dir}/")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _round_sd_1sig(x: float) -> float:
    x = float(x)
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x
    exp = math.floor(math.log10(abs(x)))
    return round(x, -exp)


def _fmt(mean: float, sd: float) -> str:
    sd1 = _round_sd_1sig(sd)
    if sd1 == 0.0 or math.isnan(sd1):
        return f"{round(float(mean), 3):.3f}$\\pm$0"
    exp   = math.floor(math.log10(abs(sd1)))
    dec   = max(0, -exp)
    mean_r = round(float(mean), dec)
    return f"{mean_r:.{dec}f}$\\pm${sd1:.{dec}f}"


def _tex(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}")
            .replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def _tabular(df: pd.DataFrame, colspec: str, caption: str, label: str) -> str:
    rows = [f"\\begin{{table}}[t]", "\\centering",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
            " & ".join(df.columns) + " \\\\", "\\hline"]
    for _, row in df.iterrows():
        rows.append(" & ".join(str(v) for v in row.values) + " \\\\")
    rows += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(rows) + "\n"


def write_latex(agg: pd.DataFrame, support_df: pd.DataFrame,
                top_fp: pd.DataFrame, top_fn: pd.DataFrame,
                latex_dir: Path, prefix: str, cfg: Config,
                top_k: int, rare_threshold: int):
    latex_dir.mkdir(parents=True, exist_ok=True)
    task_slug = f"{cfg.task}_{cfg.model}_T{cfg.temperature}"

    # Support table
    df = support_df.sort_values("support", ascending=False).reset_index(drop=True)
    head = df.head(top_k)
    rest = df.iloc[top_k:]
    rows = [[_tex(r["label"]), int(r["support"])] for _, r in head.iterrows()]
    if len(rest):
        rows.append(["Other", int(rest["support"].sum())])
    t1 = pd.DataFrame(rows, columns=["Label", "Support"])

    # PRF table
    keep = (pd.concat([agg.head(top_k), agg[agg["support"] <= rare_threshold]])
            .drop_duplicates("label")
            .sort_values(["support", "label"], ascending=[False, True])
            .reset_index(drop=True))
    t2 = pd.DataFrame([
        [_tex(r["label"]), int(r["support"]),
         _fmt(r["precision_mean"], r["precision_std"]),
         _fmt(r["recall_mean"],    r["recall_std"]),
         _fmt(r["f1_mean"],        r["f1_std"])]
        for _, r in keep.iterrows()
    ], columns=["Label", "Support", "P", "R", "F1"])

    # Top-FP / FN tables
    def _err_table(df_err, kind):
        mc = f"{kind}_mean"
        sc = f"{kind}_std"
        df_err = df_err.sort_values(mc, ascending=False).head(top_k)
        sd_vals = df_err[sc] if sc in df_err.columns else pd.Series(0.0, index=df_err.index)
        return pd.DataFrame(
            [[_tex(r["label"]), _fmt(r[mc], sd_vals.loc[r.name])] for _, r in df_err.iterrows()],
            columns=["Label", f"{kind.upper()} per run"])

    t3 = _err_table(top_fp, "fp")
    t4 = _err_table(top_fn, "fn")

    slug_display = TASK_DISPLAY.get(cfg.task, cfg.task)
    (latex_dir / f"{prefix}_support_{cfg.task}.tex").write_text(
        _tabular(t1, "lr",
                 f"Label prevalence — {slug_display}.",
                 f"tab:{task_slug}:support"), encoding="utf-8")
    (latex_dir / f"{prefix}_prf_{cfg.task}.tex").write_text(
        _tabular(t2, "lrrrr",
                 f"Per-label P/R/F1 — {slug_display}. "
                 f"Mean$\\pm$SD across runs.",
                 f"tab:{task_slug}:prf"), encoding="utf-8")
    (latex_dir / f"{prefix}_topFP_{cfg.task}.tex").write_text(
        _tabular(t3, "lr",
                 f"Top-{top_k} FP labels — {slug_display}.",
                 f"tab:{task_slug}:topFP"), encoding="utf-8")
    (latex_dir / f"{prefix}_topFN_{cfg.task}.tex").write_text(
        _tabular(t4, "lr",
                 f"Top-{top_k} FN labels — {slug_display}.",
                 f"tab:{task_slug}:topFN"), encoding="utf-8")
    print(f"  → LaTeX tables written to {latex_dir}/")


def write_latex_prf_table(agg: pd.DataFrame, latex_dir: Path,
                           prefix: str, cfg: Config,
                           top_k: int) -> None:
    # Compact booktabs table: rows = labels by support desc, cols = # | F1 mean+-SD.
    # Requires \usepackage{booktabs} in the LaTeX preamble.
    latex_dir.mkdir(parents=True, exist_ok=True)

    df = (agg.sort_values(["support", "label"], ascending=[False, True])
            .reset_index(drop=True)
            .head(top_k))

    task_display = TASK_DISPLAY.get(cfg.task, cfg.task)
    task_slug    = f"{cfg.task}_{cfg.model}_T{cfg.temperature}"

    cap = (
        "Per-label F\\textsubscript{1} for the "
        "\\textbf{" + task_display + "} task "
        "(model: " + _tex(cfg.model) + ", $T=" + cfg.temperature + "$). "
        "Rows ordered by support (\\#); "
        "values are mean\\,$\\pm$\\,SD across runs."
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{" + cap + "}",
        "\\label{tab:" + task_slug + ":f1}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Label} & \\textbf{\\#} & \\textbf{F\\textsubscript{1}} \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(
            _tex(row["label"]) + " & "
            + str(int(row["support"])) + " & "
            + _fmt(row["f1_mean"], row["f1_std"]) + " \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = latex_dir / (prefix + "_f1_table_" + cfg.task + ".tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {out}")


def write_latex_combined_prf_table(
        task_aggs: Dict[str, pd.DataFrame],
        latex_dir: Path,
        prefix: str,
        model: str,
        temperature: str,
        top_k: int,
) -> None:
    # Single full-page booktabs table stacking all tasks.
    # Structure:
    #   - one block per task, separated by a \midrule + task header row
    #   - within each block: rows = labels, sorted by support desc, capped at top_k
    #   - columns: Label | # | P | R | F1   (all mean+-SD)
    # Requires \usepackage{booktabs} and \usepackage{longtable} in preamble.
    # longtable allows the table to break across pages naturally.
    latex_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{model}_T{temperature}"
    cap  = (
        "Per-label precision, recall and F\\textsubscript{1} for all tasks "
        "(model: " + _tex(model) + ", $T=" + temperature + "$). "
        "Within each task block rows are ordered by support (\\#); "
        "values are mean\\,$\\pm$\\,SD across runs."
    )

    lines = [
        "\\begin{longtable}{llrrrr}",
        "\\caption{" + cap + "}",
        "\\label{tab:" + slug + ":combined_prf} \\\\",
        "\\toprule",
        "\\textbf{Task} & \\textbf{Label} & \\textbf{\\#}"
        " & \\textbf{P} & \\textbf{R} & \\textbf{F\\textsubscript{1}} \\\\",
        "\\midrule",
        "\\endfirsthead",
        # repeated header on continuation pages
        "\\toprule",
        "\\textbf{Task} & \\textbf{Label} & \\textbf{\\#}"
        " & \\textbf{P} & \\textbf{R} & \\textbf{F\\textsubscript{1}} \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{6}{r}{\\small\\itshape (continued on next page)} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]

    for t_idx, (task, agg) in enumerate(task_aggs.items()):
        if t_idx > 0:
            lines.append("\\midrule")          # separator between task blocks

        df = (agg.sort_values(["support", "label"], ascending=[False, True])
                .reset_index(drop=True)
                .head(top_k))

        task_display = TASK_DISPLAY.get(task, task)
        n_rows = len(df)

        for r_idx, (_, row) in enumerate(df.iterrows()):
            # Task name only on the first row of the block, using multirow
            if r_idx == 0:
                task_cell = "\\multirow{" + str(n_rows) + "}{*}{\\textbf{" + _tex(task_display) + "}}"
            else:
                task_cell = ""
            lines.append(
                task_cell + " & "
                + _tex(row["label"]) + " & "
                + str(int(row["support"])) + " & "
                + _fmt(row["precision_mean"], row["precision_std"]) + " & "
                + _fmt(row["recall_mean"],    row["recall_std"])    + " & "
                + _fmt(row["f1_mean"],        row["f1_std"])        + " \\\\"
            )

    lines.append("\\end{longtable}")

    out = latex_dir / (prefix + f"_{model}_T{temperature}_combined_prf_table.tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f" \u2192 {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_f1bar(agg: pd.DataFrame, path: Path, top_k: int, rare_threshold: int):
    keep = (pd.concat([agg.head(top_k), agg[agg["support"] <= rare_threshold]])
            .drop_duplicates("label")
            .sort_values(["support", "label"], ascending=[False, True])
            .reset_index(drop=True))
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(max(6, len(keep) * 0.7), 4))
    ax.errorbar(x, keep["f1_mean"], yerr=keep["f1_std"].fillna(0),
                fmt="o", capsize=3, color=_METRIC_STYLES["F1"]["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(keep["label"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, path)


def plot_support_bar(support_df: pd.DataFrame, path: Path, top_k: int):
    df = support_df.sort_values("support", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.7), 4))
    ax.bar(df["label"], df["support"], color="#607D8B")
    ax.set_ylabel("Support")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, path)


def plot_top_errors_bar(err_df: pd.DataFrame, path: Path, kind: str, top_k: int):
    mc = f"{kind}_mean"
    df = err_df.sort_values(mc, ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.7), 4))
    ax.bar(df["label"], df[mc],
           color=_METRIC_STYLES["Precision"]["color"] if kind == "fp"
           else _METRIC_STYLES["Recall"]["color"])
    ax.set_ylabel(f"Mean {kind.upper()} per run")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, path)


def _draw_prevalence_ax(ax: plt.Axes, agg: pd.DataFrame,
                        title: str = "",
                        annotate_top_n: int = 6,
                        log_x: bool = False,
                        title_fontsize: int = 13,
                        annot_fontsize: int = 9,
                        axis_label_fontsize: int = 11,
                        tick_fontsize: int = 10,
                        marker_size: int = 70):
    """Core drawing primitive shared by single-task and grid plots."""
    metrics = [
        ("F1",        "f1_mean",        "f1_std"),
        ("Precision", "precision_mean", "precision_std"),
        ("Recall",    "recall_mean",    "recall_std"),
    ]
    handles = []
    for name, mc, sc in metrics:
        st = _METRIC_STYLES[name]
        ax.errorbar(agg["support"], agg[mc], yerr=agg[sc].fillna(0),
                    fmt="none", ecolor=st["color"], alpha=0.35, capsize=4, zorder=2)
        h = ax.scatter(agg["support"], agg[mc], label=name,
                       color=st["color"], marker=st["marker"],
                       s=marker_size, linewidths=0.6, edgecolors="white",
                       zorder=st["zorder"])
        handles.append(h)

    for _, row in agg.head(annotate_top_n).iterrows():
        ax.annotate(row["label"], xy=(row["support"], row["f1_mean"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=annot_fontsize, alpha=0.85, clip_on=True)

    ax.set_xlabel("Prevalence (support)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Score", fontsize=axis_label_fontsize)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":",  alpha=0.25)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=10)
    return handles


def plot_prevalence(agg: pd.DataFrame, path: Path,
                    title: str = "", annotate_top_n: int = 6,
                    log_x: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    handles = _draw_prevalence_ax(ax, agg, title=title,
                                  annotate_top_n=annotate_top_n, log_x=log_x)
    ax.legend(handles=handles, labels=[h.get_label() for h in handles],
              framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_prevalence_grid(task_aggs: Dict[str, pd.DataFrame], path: Path,
                         annotate_top_n: int = 5, log_x: bool = False,
                         ncols: int = 2,
                         panel_size: Tuple[float, float] = (6.5, 5.0),
                         title_fontsize: int = 13,
                         annot_fontsize: int = 9,
                         axis_label_fontsize: int = 11,
                         tick_fontsize: int = 10,
                         legend_fontsize: int = 11,
                         marker_size: int = 70):
    tasks  = list(task_aggs.keys())
    nrows  = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    shared_handles = None
    for idx, task in enumerate(tasks):
        ri, ci = divmod(idx, ncols)
        ax = axes[ri][ci]
        handles = _draw_prevalence_ax(
            ax, task_aggs[task],
            title=TASK_DISPLAY.get(task, task),
            annotate_top_n=annotate_top_n,
            log_x=log_x,
            title_fontsize=title_fontsize,
            annot_fontsize=annot_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            marker_size=marker_size,
        )
        if shared_handles is None:
            shared_handles = handles

    for idx in range(len(tasks), nrows * ncols):
        ri, ci = divmod(idx, ncols)
        axes[ri][ci].set_visible(False)

    if shared_handles:
        fig.legend(handles=shared_handles,
                   labels=[h.get_label() for h in shared_handles],
                   loc="lower center", ncol=len(_METRIC_STYLES),
                   fontsize=legend_fontsize, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="label_analysis.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── What to analyse ──────────────────────────────────────────────────────
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                    choices=ALL_TASKS, metavar="TASK",
                    help=f"Task(s) to process. Choices: {ALL_TASKS}. "
                         f"Default: all tasks.")
    ap.add_argument("--model", required=True,
                    help="Model name, e.g. gemini-2-5-pro")
    ap.add_argument("--temperature", required=True,
                    help="Temperature string, e.g. 0.0")

    # ── Input paths ──────────────────────────────────────────────────────────
    ap.add_argument("--gt", type=Path, default=Path("../sampled_papers_full.csv"),
                    help="Ground-truth TSV (default: ../sampled_papers_full.csv)")
    ap.add_argument("--pred_dir", type=Path, default=Path("../output"),
                    help="Root directory of prediction files "
                         "(default: ../output/). Ignored with --from_csv.")
    ap.add_argument("--from_csv", action="store_true",
                    help="Skip computation; load previously saved CSVs from "
                         "--out_dir and only render outputs.")

    # ── Output paths ─────────────────────────────────────────────────────────
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts"),
                    help="Root output directory (default: artifacts/). "
                         "CSVs → <out_dir>/csv/, plots → <out_dir>/plots/, "
                         "LaTeX → <out_dir>/latex/.")
    ap.add_argument("--prefix", default="lbl",
                    help="Filename prefix for all outputs (default: lbl).")

    # ── What to emit ─────────────────────────────────────────────────────────
    ap.add_argument("--emit", nargs="+", default=["csv", "plots"],
                    choices=["csv", "plots", "latex"],
                    help="Output types to produce (default: csv plots).")
    ap.add_argument("--plot_types", nargs="+", default=ALL_PLOT_TYPES,
                    choices=ALL_PLOT_TYPES, metavar="TYPE",
                    help=f"Which plots to render (default: all). "
                         f"Choices: {ALL_PLOT_TYPES}.")

    # ── Plot options ─────────────────────────────────────────────────────────
    ap.add_argument("--top_k", type=int, default=10,
                    help="Top-K labels for tables, error bars, etc. (default: 10).")
    ap.add_argument("--rare_threshold", type=int, default=5,
                    help="Labels with support ≤ this are always included in "
                         "PRF tables (default: 5).")
    ap.add_argument("--annotate_top_n", type=int, default=6,
                    help="Labels to annotate in prevalence plots (default: 6).")
    ap.add_argument("--log_x", action="store_true",
                    help="Log scale on X axis of prevalence plots.")
    ap.add_argument("--ncols", type=int, default=2,
                    help="Columns in the prevalence grid (default: 2).")

    # ── Grid font / size options ──────────────────────────────────────────────
    ap.add_argument("--grid_panel_w",   type=float, default=6.5,
                    help="Width of each panel in the grid in inches (default: 6.5).")
    ap.add_argument("--grid_panel_h",   type=float, default=5.0,
                    help="Height of each panel in the grid in inches (default: 5.0).")
    ap.add_argument("--grid_title_fs",  type=int, default=13,
                    help="Panel title font size in the grid (default: 13).")
    ap.add_argument("--grid_annot_fs",  type=int, default=9,
                    help="Label annotation font size in the grid (default: 9).")
    ap.add_argument("--grid_axis_fs",   type=int, default=11,
                    help="Axis label font size in the grid (default: 11).")
    ap.add_argument("--grid_tick_fs",   type=int, default=10,
                    help="Tick label font size in the grid (default: 10).")
    ap.add_argument("--grid_legend_fs", type=int, default=11,
                    help="Legend font size in the grid (default: 11).")
    ap.add_argument("--grid_marker_s",  type=int, default=70,
                    help="Marker size (pts^2) in the grid (default: 70).")

    return ap


def main():
    args = build_parser().parse_args()

    csv_dir   = args.out_dir / "csv"
    plot_dir  = args.out_dir / "plots"
    latex_dir = args.out_dir / "latex"

    # ── Load GT once (only needed if computing from predictions) ─────────────
    gt = None
    if not args.from_csv and "csv" in args.emit:
        gt = load_gt(args.gt)

    # ── Per-task loop ────────────────────────────────────────────────────────
    task_aggs: Dict[str, pd.DataFrame] = {}  # collected for prevalence_grid

    for task in args.tasks:
        cfg = Config(task=task, model=args.model, temperature=args.temperature)
        print(f"\n[{task}]")

        # 1) Compute or load per-label CSV
        if args.from_csv:
            p = _csv_per_label(csv_dir, args.prefix, cfg)
            if not p.exists():
                print(f"  [WARN] CSV not found, skipping: {p}")
                continue
            agg = pd.read_csv(p)
            # Normalise sd→std
            for base in ["precision", "recall", "f1"]:
                if f"{base}_sd" in agg.columns and f"{base}_std" not in agg.columns:
                    agg = agg.rename(columns={f"{base}_sd": f"{base}_std"})
        elif "csv" in args.emit:
            agg = compute_and_save(gt, cfg, args.pred_dir,
                                   csv_dir, args.prefix, args.top_k)
        else:
            print("  [SKIP] Neither --from_csv nor csv in --emit; nothing to compute.")
            continue

        task_aggs[task] = agg

        # 2) LaTeX
        if "latex" in args.emit:
            support_df = pd.read_csv(_csv_support(csv_dir, args.prefix, cfg)) \
                if _csv_support(csv_dir, args.prefix, cfg).exists() \
                else label_support([])  # fallback empty
            top_fp = pd.read_csv(_csv_topfp(csv_dir, args.prefix, cfg)) \
                if _csv_topfp(csv_dir, args.prefix, cfg).exists() else pd.DataFrame()
            top_fn = pd.read_csv(_csv_topfn(csv_dir, args.prefix, cfg)) \
                if _csv_topfn(csv_dir, args.prefix, cfg).exists() else pd.DataFrame()
            if not top_fp.empty and not top_fn.empty:
                write_latex(agg, support_df, top_fp, top_fn,
                            latex_dir, args.prefix, cfg,
                            args.top_k, args.rare_threshold)
                write_latex_prf_table(agg, latex_dir, args.prefix, cfg,
                                      args.top_k)

        # 3) Per-task plots (all except prevalence_grid)
        if "plots" in args.emit:
            pt = set(args.plot_types)
            slug = f"{args.prefix}_{task}_{args.model}_T{args.temperature}"

            if "f1bar" in pt:
                plot_f1bar(agg, plot_dir / f"{slug}_f1bar.png",
                           args.top_k, args.rare_threshold)

            if "support" in pt:
                sp = _csv_support(csv_dir, args.prefix, cfg)
                if sp.exists():
                    plot_support_bar(pd.read_csv(sp),
                                     plot_dir / f"{args.prefix}_{task}_support.png",
                                     args.top_k)

            if "top_errors" in pt:
                fp_p = _csv_topfp(csv_dir, args.prefix, cfg)
                fn_p = _csv_topfn(csv_dir, args.prefix, cfg)
                if fp_p.exists():
                    plot_top_errors_bar(pd.read_csv(fp_p),
                                        plot_dir / f"{slug}_topFP.png", "fp", args.top_k)
                if fn_p.exists():
                    plot_top_errors_bar(pd.read_csv(fn_p),
                                        plot_dir / f"{slug}_topFN.png", "fn", args.top_k)

            if "prevalence" in pt:
                plot_prevalence(
                    agg,
                    path=plot_dir / f"{slug}_prevalence.png",
                    title=TASK_DISPLAY.get(task, task),
                    annotate_top_n=args.annotate_top_n,
                    log_x=args.log_x,
                )

    # ── Combined prevalence grid (needs all tasks collected) ─────────────────
    if "plots" in args.emit and "prevalence_grid" in args.plot_types and task_aggs:
        grid_slug = f"{args.prefix}_{args.model}_T{args.temperature}"
        plot_prevalence_grid(
            task_aggs,
            path=plot_dir / f"{grid_slug}_prevalence_grid.png",
            annotate_top_n=args.annotate_top_n,
            log_x=args.log_x,
            ncols=args.ncols,
            panel_size=(args.grid_panel_w, args.grid_panel_h),
            title_fontsize=args.grid_title_fs,
            annot_fontsize=args.grid_annot_fs,
            axis_label_fontsize=args.grid_axis_fs,
            tick_fontsize=args.grid_tick_fs,
            legend_fontsize=args.grid_legend_fs,
            marker_size=args.grid_marker_s,
        )

    # ── Combined PRF table (needs all tasks collected) ───────────────────────
    if "latex" in args.emit and task_aggs:
        write_latex_combined_prf_table(
            task_aggs,
            latex_dir,
            prefix=args.prefix,
            model=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
