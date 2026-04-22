#!/usr/bin/env python3
"""
label_artifacts_from_csv.py

Generate compact LaTeX tables and/or plots from analysis CSVs produced by:
  - label_report.py (preferred)
  - older label_error_breakdown.py variants

This script is schema-tolerant (sd/std naming) and is useful when you already
computed the CSVs and want to re-render tables/figures.

Single-task usage:
  python label_artifacts_from_csv.py \
    --support support.csv \
    --per_label per_label.csv \
    --top_fp top_fp.csv \
    --top_fn top_fn.csv \
    --emit latex plots \
    --out_dir artifacts \
    --top_k 8 --rare_threshold 5 \
    --prefix geo_pro_T0

Multi-task combined prevalence plot:
  python label_artifacts_from_csv.py \
    --multi_task geo:geo_per_label.csv data-type:dtype_per_label.csv \
    --emit plots \
    --out_dir artifacts \
    --prefix all_tasks
  (--support / --top_fp / --top_fn are not required in multi-task-only mode)


 python label_artifacts_from_csv.py \
  --multi_task \
    geo:label_analysis_per_label_geo_gemini-2-5-pro_T0.0.csv \
    data-type:label_analysis_per_label_data-type_gemini-2-5-pro_T0.0.csv \
    data-accessibility:label_analysis_per_label_data-accessibility_gemini-2-5-pro_T0.0.csv \
    paper-type:label_analysis_per_label_paper-type_gemini-2-5-pro_T0.0.csv \
  --emit plots \
  --out_dir artifacts \
  --prefix all_tasks

  """

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Aesthetic constants
# ─────────────────────────────────────────────────────────────

_METRIC_STYLES: Dict[str, dict] = {
    "F1":        {"color": "#2196F3", "marker": "o", "zorder": 4},
    "Precision": {"color": "#E91E63", "marker": "s", "zorder": 3},
    "Recall":    {"color": "#4CAF50", "marker": "^", "zorder": 3},
}

_TASK_DISPLAY = {
    "geo":                "Geography",
    "data-type":          "Data Type",
    "data-accessibility": "Data Accessibility",
    "paper-type":         "Paper Type",
}

# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def round_sd_1sig(x: float) -> float:
    if x is None:
        return float("nan")
    x = float(x)
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x
    exp = math.floor(math.log10(abs(x)))
    return round(x, -exp)


def round_mean_to_sd(mean: float, sd_1sig: float) -> float:
    if sd_1sig == 0.0 or math.isnan(sd_1sig):
        return round(float(mean), 3)
    exp = math.floor(math.log10(abs(sd_1sig)))
    return round(float(mean), max(0, -exp))


def fmt_mean_pm_sd(mean: float, sd: float) -> str:
    sd1 = round_sd_1sig(sd)
    mean_r = round_mean_to_sd(mean, sd1)
    if sd1 == 0.0 or math.isnan(sd1):
        return f"{mean_r:.3f}$\\pm$0"
    exp = math.floor(math.log10(abs(sd1))) if sd1 != 0 else 0
    decimals = max(0, -exp)
    return f"{mean_r:.{decimals}f}$\\pm${sd1:.{decimals}f}"


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def df_to_tabular(df: pd.DataFrame, colspec: str, caption: str, label: str) -> str:
    lines: List[str] = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(df.columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(x) for x in row.values) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _pick(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found. Columns: {list(df.columns)}")


def _normalise_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """Rename *_sd → *_std if needed; return a copy."""
    df = df.copy()
    for base in ["precision", "recall", "f1", "tp", "fp", "fn"]:
        if f"{base}_sd" in df.columns and f"{base}_std" not in df.columns:
            df = df.rename(columns={f"{base}_sd": f"{base}_std"})
    return df


# ─────────────────────────────────────────────────────────────
# Compact table builders
# ─────────────────────────────────────────────────────────────

def compact_support(support_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    label_col = _pick(support_df, ["label", "Label"])
    sup_col   = _pick(support_df, ["support", "Support", "count", "n"])
    df = (support_df[[label_col, sup_col]]
          .rename(columns={label_col: "label", sup_col: "support"})
          .sort_values("support", ascending=False)
          .reset_index(drop=True))
    head = df.head(top_k)
    rest = df.iloc[top_k:]
    rows = [[tex_escape(r["label"]), int(r["support"])] for _, r in head.iterrows()]
    if len(rest):
        rows.append(["Other", int(rest["support"].sum())])
    return pd.DataFrame(rows, columns=["Label", "Support"])


def compact_per_label(per_df: pd.DataFrame, top_k: int, rare_threshold: int) -> pd.DataFrame:
    per_df = _normalise_per_label(per_df)
    label_col   = _pick(per_df, ["label", "Label"])
    support_col = _pick(per_df, ["support", "Support"])
    df = per_df.rename(columns={label_col: "label", support_col: "support"})

    required = ["label", "support",
                "precision_mean", "precision_std",
                "recall_mean",    "recall_std",
                "f1_mean",        "f1_std"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns: {list(df.columns)}")

    df = df.sort_values("support", ascending=False).reset_index(drop=True)
    keep = (pd.concat([df.head(top_k), df[df["support"] <= rare_threshold]])
              .drop_duplicates(subset=["label"])
              .sort_values(["support", "label"], ascending=[False, True])
              .reset_index(drop=True))

    rows = [
        [tex_escape(r["label"]), int(r["support"]),
         fmt_mean_pm_sd(r["precision_mean"], r["precision_std"]),
         fmt_mean_pm_sd(r["recall_mean"],    r["recall_std"]),
         fmt_mean_pm_sd(r["f1_mean"],        r["f1_std"])]
        for _, r in keep.iterrows()
    ]
    return pd.DataFrame(rows, columns=["Label", "Support", "P", "R", "F1"])


def compact_top_errors(err_df: pd.DataFrame, kind: str, top_k: int) -> pd.DataFrame:
    label_col = _pick(err_df, ["label", "Label"])
    kind_l    = kind.lower()
    mean_col  = _pick(err_df, [f"{kind_l}_mean", f"{kind_l}_avg", kind_l])
    sd_col    = _pick(err_df, [f"{kind_l}_std",  f"{kind_l}_sd",  f"{kind_l}_sdev"])

    df = err_df.rename(columns={label_col: "label",
                                 mean_col:  f"{kind_l}_mean",
                                 sd_col:    f"{kind_l}_std"}).copy()
    df = df.sort_values(f"{kind_l}_mean", ascending=False).head(top_k).reset_index(drop=True)
    rows = [
        [tex_escape(r["label"]),
         fmt_mean_pm_sd(r[f"{kind_l}_mean"], r[f"{kind_l}_std"])]
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows, columns=["Label", f"{kind} per run"])


# ─────────────────────────────────────────────────────────────
# Classic plots (unchanged from original)
# ─────────────────────────────────────────────────────────────

def plot_support(df: pd.DataFrame, out: Path, top_k: int):
    fig, ax = plt.subplots()
    ax.bar(df["Label"], df["Support"])
    ax.set_ylabel("Support (count)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_per_label_f1_from_raw(per_df: pd.DataFrame, out: Path,
                                top_k: int, rare_threshold: int):
    per_df = _normalise_per_label(per_df)
    df = per_df.sort_values("support", ascending=False).reset_index(drop=True)
    keep = (pd.concat([df.head(top_k), df[df["support"] <= rare_threshold]])
              .drop_duplicates(subset=["label"])
              .sort_values(["support", "label"], ascending=[False, True])
              .reset_index(drop=True))

    x = np.arange(len(keep))
    fig, ax = plt.subplots()
    ax.errorbar(x, keep["f1_mean"],
                yerr=keep.get("f1_std", pd.Series(0.0, index=keep.index)),
                fmt="o", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keep["label"], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (per label)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_top_errors(err_df: pd.DataFrame, out: Path, kind: str, top_k: int):
    mean_col = _pick(err_df, [f"{kind.lower()}_mean", f"{kind.lower()}_avg", kind.lower()])
    df = err_df.sort_values(mean_col, ascending=False).head(top_k).reset_index(drop=True)
    fig, ax = plt.subplots()
    ax.bar(df["label"], df[mean_col])
    ax.set_ylabel(f"{kind} per run (mean)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# NEW: prevalence vs metric scatter
# ─────────────────────────────────────────────────────────────

def _prep_prevalence_data(per_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy DataFrame sorted by support descending, with columns:
      label, support, f1_mean, f1_std, precision_mean, precision_std,
      recall_mean, recall_std
    """
    per_df = _normalise_per_label(per_df)
    label_col   = _pick(per_df, ["label", "Label"])
    support_col = _pick(per_df, ["support", "Support"])
    df = per_df.rename(columns={label_col: "label", support_col: "support"})

    required = ["label", "support",
                "f1_mean",        "f1_std",
                "precision_mean", "precision_std",
                "recall_mean",    "recall_std"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for prevalence plot: {missing}")

    return df[required].sort_values("support", ascending=False).reset_index(drop=True)


def _draw_prevalence_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str = "",
    annotate_top_n: int = 6,
    log_x: bool = False,
    title_fontsize: int = 11,
    label_fontsize: int = 7,
) -> List:
    """
    Draw the prevalence-vs-metrics scatter onto *ax* and return legend handles.
    Separated from figure creation so it can be reused inside a subplot grid.
    """
    metrics = [
        ("F1",        "f1_mean",        "f1_std"),
        ("Precision", "precision_mean", "precision_std"),
        ("Recall",    "recall_mean",    "recall_std"),
    ]

    handles = []
    for metric_name, mean_col, std_col in metrics:
        style = _METRIC_STYLES[metric_name]
        yerr  = df[std_col].fillna(0).values
        ax.errorbar(
            df["support"], df[mean_col],
            yerr=yerr,
            fmt="none", ecolor=style["color"], alpha=0.35,
            capsize=3, zorder=2,
        )
        sc = ax.scatter(
            df["support"], df[mean_col],
            label=metric_name,
            color=style["color"],
            marker=style["marker"],
            s=55, linewidths=0.6,
            edgecolors="white",
            zorder=style["zorder"],
        )
        handles.append(sc)

    # Annotate top-N labels at their F1 position
    if annotate_top_n > 0:
        for _, row in df.head(annotate_top_n).iterrows():
            ax.annotate(
                row["label"],
                xy=(row["support"], row["f1_mean"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=label_fontsize, alpha=0.8,
                clip_on=True,
            )

    ax.set_xlabel("Prevalence (support)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=8)

    return handles


def plot_prevalence_vs_metrics(
    per_df: pd.DataFrame,
    out: Path,
    title: str = "",
    annotate_top_n: int = 6,
    log_x: bool = False,
    figsize: Tuple[float, float] = (7, 4.5),
) -> None:
    """
    Scatter plot: X = label prevalence (support), Y = P / R / F1.

    Each metric is drawn as a separate series with a distinct color + marker.
    Error bars show ±1 SD across repeated runs.
    The top-N labels by support are annotated with their name.

    Parameters
    ----------
    per_df        : raw per-label CSV loaded via pd.read_csv
    out           : output file path (.png / .pdf)
    title         : figure title
    annotate_top_n: how many labels to annotate (highest support first)
    log_x         : use log scale on X axis (helpful for skewed support distributions)
    figsize       : figure size in inches (width, height)
    """
    df = _prep_prevalence_data(per_df)

    fig, ax = plt.subplots(figsize=figsize)
    handles = _draw_prevalence_ax(
        ax, df,
        title=title,
        annotate_top_n=annotate_top_n,
        log_x=log_x,
    )
    ax.legend(handles=handles,
              labels=[h.get_label() for h in handles],
              framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote prevalence plot: {out}")


def plot_multi_task_prevalence_vs_metrics(
    task_data: Dict[str, pd.DataFrame],
    out: Path,
    annotate_top_n: int = 5,
    log_x: bool = False,
    figsize_per_panel: Tuple[float, float] = (5.5, 4.0),
    ncols: int = 2,
) -> None:
    """
    Grid figure: one panel per task, each showing prevalence vs P/R/F1.

    A shared legend is placed below all panels.

    Parameters
    ----------
    task_data         : {task_name: per_label_df} — pass raw CSVs, not compacted tables
    out               : output file path (.png / .pdf)
    annotate_top_n    : labels to annotate per panel
    log_x             : log scale on X axis
    figsize_per_panel : (width, height) per subplot in inches
    ncols             : columns in the panel grid
    """
    tasks = list(task_data.keys())
    n     = len(tasks)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    shared_handles: Optional[List] = None

    for idx, task_name in enumerate(tasks):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]

        df = _prep_prevalence_data(task_data[task_name])
        display_title = _TASK_DISPLAY.get(task_name, task_name)

        handles = _draw_prevalence_ax(
            ax, df,
            title=display_title,
            annotate_top_n=annotate_top_n,
            log_x=log_x,
            title_fontsize=10,
            label_fontsize=6.5,
        )
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)

        if shared_handles is None:
            shared_handles = handles

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i][col_i].set_visible(False)

    # Shared legend below the grid
    if shared_handles:
        fig.legend(
            handles=shared_handles,
            labels=[h.get_label() for h in shared_handles],
            loc="lower center",
            ncol=len(_METRIC_STYLES),
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote multi-task prevalence grid: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate LaTeX tables and/or plots from label-analysis CSVs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── single-task inputs ──────────────────────────────────
    ap.add_argument("--support",   default=None, help="Support CSV (single task)")
    ap.add_argument("--per_label", default=None, help="Per-label metrics CSV (single task)")
    ap.add_argument("--top_fp",    default=None, help="Top-FP CSV (single task)")
    ap.add_argument("--top_fn",    default=None, help="Top-FN CSV (single task)")
    ap.add_argument("--confusion", default=None, help="Confusion matrix CSV (optional)")

    # ── multi-task prevalence input ─────────────────────────
    ap.add_argument(
        "--multi_task", nargs="+", default=None,
        metavar="TASK:PER_LABEL_CSV",
        help=(
            "One or more TASK:CSV pairs for the combined prevalence grid.\n"
            "Example: --multi_task geo:geo_pl.csv data-type:dtype_pl.csv\n"
            "Can be combined with --per_label or used on its own."
        ),
    )

    # ── output / rendering options ──────────────────────────
    ap.add_argument("--emit", nargs="+", default=["latex"],
                    choices=["latex", "plots"])
    ap.add_argument("--out_dir",        default="artifacts")
    ap.add_argument("--prefix",         default="task",
                    help="Prefix for all output filenames (and LaTeX label ids).")
    ap.add_argument("--top_k",          type=int, default=8)
    ap.add_argument("--rare_threshold", type=int, default=5)
    ap.add_argument("--annotate_top_n", type=int, default=6,
                    help="Number of labels to annotate in prevalence plots.")
    ap.add_argument("--log_x",          action="store_true",
                    help="Use log scale on the X axis of prevalence plots.")
    ap.add_argument("--ncols",          type=int, default=2,
                    help="Number of columns in the multi-task grid.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── single-task block ───────────────────────────────────
    if args.per_label:
        per_df = pd.read_csv(args.per_label)

        if "latex" in args.emit:
            if not (args.support and args.top_fp and args.top_fn):
                print("[WARN] --support / --top_fp / --top_fn needed for LaTeX; skipping tables.")
            else:
                support_df = pd.read_csv(args.support)
                fp_df      = pd.read_csv(args.top_fp)
                fn_df      = pd.read_csv(args.top_fn)

                t1 = compact_support(support_df, args.top_k)
                t2 = compact_per_label(per_df, args.top_k, args.rare_threshold)
                t3 = compact_top_errors(fp_df, "FP", args.top_k)
                t4 = compact_top_errors(fn_df, "FN", args.top_k)

                (out_dir / "label_support.tex").write_text(
                    df_to_tabular(
                        t1, "lr",
                        "Label prevalence in the evaluation set "
                        "(top labels; remaining aggregated as Other).",
                        f"tab:{args.prefix}:label_support"),
                    encoding="utf-8")
                (out_dir / "per_label_prf.tex").write_text(
                    df_to_tabular(
                        t2, "lrrrr",
                        "Per-label performance (top labels by support and all rare labels). "
                        "Values are mean$\\pm$SD across runs.",
                        f"tab:{args.prefix}:per_label_prf"),
                    encoding="utf-8")
                (out_dir / "top_fp.tex").write_text(
                    df_to_tabular(
                        t3, "lr",
                        "Labels with the highest false-positive frequency "
                        "(mean$\\pm$SD across runs).",
                        f"tab:{args.prefix}:top_fp"),
                    encoding="utf-8")
                (out_dir / "top_fn.tex").write_text(
                    df_to_tabular(
                        t4, "lr",
                        "Labels with the highest false-negative frequency "
                        "(mean$\\pm$SD across runs).",
                        f"tab:{args.prefix}:top_fn"),
                    encoding="utf-8")
                print(f"[OK] Wrote LaTeX tables to: {out_dir}")

        if "plots" in args.emit:
            if args.support:
                t1 = compact_support(pd.read_csv(args.support), args.top_k)
                plot_support(t1, out_dir / "support.png", args.top_k)

            plot_per_label_f1_from_raw(
                per_df, out_dir / "per_label_f1.png",
                args.top_k, args.rare_threshold)

            if args.top_fp:
                plot_top_errors(pd.read_csv(args.top_fp),
                                out_dir / "top_fp.png", "FP", args.top_k)
            if args.top_fn:
                plot_top_errors(pd.read_csv(args.top_fn),
                                out_dir / "top_fn.png", "FN", args.top_k)

            # Single-task prevalence scatter
            plot_prevalence_vs_metrics(
                per_df,
                out=out_dir / f"{args.prefix}_prevalence_vs_metrics.png",
                title=_TASK_DISPLAY.get(args.prefix, args.prefix),
                annotate_top_n=args.annotate_top_n,
                log_x=args.log_x,
            )

    # ── multi-task combined prevalence grid ─────────────────
    if "plots" in args.emit and args.multi_task:
        task_data: Dict[str, pd.DataFrame] = {}
        for item in args.multi_task:
            if ":" not in item:
                raise ValueError(
                    f"--multi_task entries must be TASK:CSV, got: {item!r}")
            task_name, csv_path = item.split(":", 1)
            task_data[task_name] = pd.read_csv(csv_path)

        plot_multi_task_prevalence_vs_metrics(
            task_data,
            out=out_dir / f"{args.prefix}_prevalence_grid.png",
            annotate_top_n=args.annotate_top_n,
            log_x=args.log_x,
            ncols=args.ncols,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
