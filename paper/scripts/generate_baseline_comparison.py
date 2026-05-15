"""
Generate baseline comparison figures and tables for the EpiScope paper.

Outputs (relative to repo root):
  paper/figures/fig_baseline_comparison.pdf    -- main-text grouped dot chart
  paper/figures/fig_corpus_audit.pdf           -- appendix unsupervised heatmap
  paper/tables/tab_baseline_comparison.tex     -- appendix full comparison table

Usage:
    python paper/scripts/generate_baseline_comparison.py [--repo-root PATH]

Inputs read automatically:
    eval_outputs/classification/summary_metrics.csv
    eval_outputs/classification_baselines/summary_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Colour palette (family-coded)
# ---------------------------------------------------------------------------
COLOURS = {
    "unsupervised": "#9ecae1",   # light blue
    "zero_shot":    "#74c476",   # green
    "supervised":   "#fd8d3c",   # orange
    "episc":        "#d62728",   # red
}

TASK_LABELS = {
    "paper-type":        "Paper type",
    "geo":               "Geography",
    "data-type":         "Data type",
    "data-accessibility": "Data access.",
}
TASK_ORDER = ["paper-type", "geo", "data-type", "data-accessibility"]

# EpiScope numbers (from tab_metrics_summary.tex in the paper)
EPISC = {
    "paper-type":         {"flash_t0": (0.877, 0.010), "pro_t0": (0.903, 0.008)},
    "geo":                {"flash_t0": (0.815, 0.009), "pro_t0": (0.888, 0.006)},
    "data-type":          {"flash_t0": (0.848, 0.007), "pro_t0": (0.900, 0.020)},
    "data-accessibility": {"flash_t0": (0.570, 0.020), "pro_t0": (0.760, 0.040)},
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_csv(repo_root / "eval_outputs/classification/summary_metrics.csv")
    df_kfold = pd.read_csv(repo_root / "eval_outputs/classification_baselines/summary_metrics.csv")
    return df_full, df_kfold


def _best_j(df: pd.DataFrame, task: str, model_prefix: str) -> float:
    mask = (df["task"] == task) & (df["model"].str.startswith(model_prefix))
    vals = df.loc[mask, "jaccard_samples_mean"]
    return float(vals.max()) if len(vals) else float("nan")


def _exact_j(df: pd.DataFrame, task: str, model: str) -> float:
    mask = (df["task"] == task) & (df["model"] == model)
    vals = df.loc[mask, "jaccard_samples_mean"]
    return float(vals.iloc[0]) if len(vals) else float("nan")


def build_comparison_table(df_full: pd.DataFrame, df_kfold: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy DataFrame: rows = methods, columns = tasks."""
    records: list[dict] = []

    for task in TASK_ORDER:
        rec: dict = {"task": task}

        # --- Zero-shot ---
        rec["majority"]       = _exact_j(df_full, task, "majority")
        rec["proto_tfidf"]    = _exact_j(df_kfold, task, "prototype_similarity")
        rec["proto_specter"]  = _exact_j(df_full, task, "prototype_similarity-emb_allenai-specter")
        rec["guided_bertopic"]= _best_j(df_full,  task, "bertopic_guided")

        # --- Supervised ---
        rec["sup_lr"]         = _exact_j(df_kfold, task, "supervised_tfidf_logreg-cv_kfold_5")
        rec["sup_svm"]        = _exact_j(df_kfold, task, "supervised_tfidf_linear_svm-cv_kfold_5")

        # --- Unsupervised (best K) ---
        for method in ("lsa", "lda", "nmf", "plsa", "bertopic"):
            rec[f"unsup_{method}"] = _best_j(df_full, task, method)

        # --- EpiScope ---
        rec["episc_flash"] = EPISC[task]["flash_t0"][0]
        rec["episc_flash_std"] = EPISC[task]["flash_t0"][1]
        rec["episc_pro"]   = EPISC[task]["pro_t0"][0]
        rec["episc_pro_std"] = EPISC[task]["pro_t0"][1]

        records.append(rec)

    return pd.DataFrame(records).set_index("task")


# ---------------------------------------------------------------------------
# Figure 1: Grouped dot chart (main text)
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped horizontal dot chart: one panel per task, methods on y-axis."""

    methods = [
        # (column, display label, family)
        ("majority",        "Majority prior",           "zero_shot"),
        ("guided_bertopic", "Guided BERTopic (best K)", "zero_shot"),
        ("proto_tfidf",     "Prototype (TF-IDF)",       "zero_shot"),
        ("sup_svm",         "TF-IDF + SVM (5-fold CV)", "supervised"),
        ("episc_flash",     "EpiScope flash, $T{=}0$",  "episc"),
        ("episc_pro",       "EpiScope pro, $T{=}0$",    "episc"),
    ]

    n_tasks = len(TASK_ORDER)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6.8, 2.6), sharey=True)

    for ax, task in zip(axes, TASK_ORDER):
        row = df.loc[task]
        y_positions = list(range(len(methods)))

        for yi, (col, label, family) in enumerate(methods):
            val = row[col]
            color = COLOURS[family]
            marker = "o"
            markersize = 7

            # Error bar only for EpiScope
            if col.startswith("episc_"):
                std_col = col + "_std"
                std = row[std_col] if std_col in row.index else 0.0
                ax.errorbar(
                    val, yi,
                    xerr=std,
                    fmt=marker,
                    color=color,
                    markersize=markersize,
                    capsize=3,
                    linewidth=1.2,
                    zorder=3,
                )
            else:
                ax.plot(val, yi, marker, color=color,
                        markersize=markersize, zorder=3)

            # Horizontal guide line
            ax.hlines(yi, 0, val, colors=color, linewidth=0.8,
                      linestyle="--", alpha=0.5, zorder=2)

        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
        ax.set_title(TASK_LABELS[task], fontsize=9, pad=4)
        ax.set_yticks(y_positions)
        ax.grid(axis="x", linewidth=0.4, alpha=0.4, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Y-axis labels only on leftmost panel
    axes[0].set_yticklabels(
        [label for _, label, _ in methods],
        fontsize=8,
    )
    axes[0].set_xlabel("Jaccard similarity", fontsize=8)

    # Shared x-axis label
    fig.text(0.52, -0.02, "Jaccard similarity $J$", ha="center", fontsize=9)

    # Legend
    handles = [
        mpatches.Patch(color=COLOURS["zero_shot"],  label="Zero-shot"),
        mpatches.Patch(color=COLOURS["supervised"], label="Supervised"),
        mpatches.Patch(color=COLOURS["episc"],      label="EpiScope (RAG)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.14),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap for corpus structure audit (appendix)
# ---------------------------------------------------------------------------

def plot_corpus_audit(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: unsupervised methods (rows) × tasks (cols), best J across K."""

    methods_display = {
        "unsup_lsa":     "LSA",
        "unsup_lda":     "LDA",
        "unsup_nmf":     "NMF",
        "unsup_plsa":    "PLSA",
        "unsup_bertopic":"BERTopic",
    }
    method_keys = list(methods_display.keys())
    task_cols = TASK_ORDER

    mat = np.array([
        [df.loc[task, mk] for task in task_cols]
        for mk in method_keys
    ])

    # Add majority row for visual reference
    majority_row = np.array([df.loc[task, "majority"] for task in task_cols])
    majority_key = "majority"
    plot_mat = np.vstack([majority_row, mat])
    row_labels = ["Majority\n(reference)"] + [methods_display[m] for m in method_keys]

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    im = ax.imshow(plot_mat, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(task_cols)):
            v = plot_mat[i, j]
            text_color = "black" if v < 0.70 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=text_color)

    ax.set_xticks(range(len(task_cols)))
    ax.set_xticklabels([TASK_LABELS[t] for t in task_cols], fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title(
        "Best Jaccard across $K$ sweep (unsupervised topic models)",
        fontsize=9, pad=6,
    )

    # Horizontal separator between reference row and topic models
    ax.axhline(0.5, color="white", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Jaccard $J$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table (appendix)
# ---------------------------------------------------------------------------

def build_latex_table(df: pd.DataFrame) -> str:
    task_cols = TASK_ORDER
    col_heads = " & ".join(r"\textbf{" + TASK_LABELS[t] + "}" for t in task_cols)

    def fmt(val: float) -> str:
        if np.isnan(val):
            return "---"
        return f"{val:.3f}"

    def episc_fmt(val: float, std: float) -> str:
        if np.isnan(val):
            return "---"
        return f"{val:.3f} $\\pm$ {std:.3f}"

    rows = []

    # ---------- Zero-shot ----------
    def row(label: str, values: list[str]) -> str:
        return label + " & " + " & ".join(values) + r" \\"

    rows.append(r"\multicolumn{5}{l}{\textit{Zero-shot baselines}} \\")
    rows.append(row(r"\quad Majority prior",
                    [fmt(df.loc[t, "majority"]) for t in task_cols]))
    rows.append(row(r"\quad Prototype similarity (TF-IDF)",
                    [fmt(df.loc[t, "proto_tfidf"]) for t in task_cols]))
    rows.append(row(r"\quad Prototype similarity (SPECTER)",
                    [fmt(df.loc[t, "proto_specter"]) for t in task_cols]))
    rows.append(row(r"\quad Guided BERTopic (best $K$)",
                    [fmt(df.loc[t, "guided_bertopic"]) for t in task_cols]))

    # ---------- Supervised ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Supervised baselines (5-fold CV)}} \\")
    rows.append(row(r"\quad TF-IDF + Logistic Regression",
                    [fmt(df.loc[t, "sup_lr"]) for t in task_cols]))
    rows.append(row(r"\quad TF-IDF + Linear SVM",
                    [fmt(df.loc[t, "sup_svm"]) for t in task_cols]))

    # ---------- Unsupervised (audit) ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Unsupervised audit (best $K$ across sweep)}} \\")
    for mkey, mname in [("unsup_lsa","LSA"), ("unsup_lda","LDA"),
                         ("unsup_nmf","NMF"), ("unsup_plsa","PLSA"),
                         ("unsup_bertopic","BERTopic")]:
        rows.append(row(r"\quad " + mname,
                        [fmt(df.loc[t, mkey]) for t in task_cols]))

    # ---------- EpiScope ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{EpiScope (RAG)}} \\")
    rows.append(row(
        r"\quad \texttt{gemini-2.5-flash}, $T{=}0$",
        [episc_fmt(df.loc[t, "episc_flash"], df.loc[t, "episc_flash_std"]) for t in task_cols],
    ))
    rows.append(row(
        r"\quad \texttt{gemini-2.5-pro}, $T{=}0$",
        [episc_fmt(df.loc[t, "episc_pro"], df.loc[t, "episc_pro_std"]) for t in task_cols],
    ))

    body = "\n".join("    " + r for r in rows)

    return rf"""\begin{{table}}[!htbp]
\centering
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Method}} & {col_heads} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{
Sample-averaged Jaccard similarity $J$ for all comparison baselines and \toolname{{}}.
Unsupervised topic-model results report the best $J$ achieved across the $K$ sweep
($K \in \{{0.5,\,1,\,2,\,4\}} \times |\mathcal{{L}}|$); they are included as a corpus structure
audit rather than as competitive classifiers (see \Cref{{app:baselines}}).
EpiScope values are mean $\pm$ standard deviation over $R{{=}}5$ repeated runs.
Supervised baselines use 5-fold cross-validation.
}}
\label{{tab:baseline_comparison}}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to the repository root. Defaults to two levels above this script.",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[2]

    print(f"Repo root: {repo_root}")

    df_full, df_kfold = load_data(repo_root)
    df = build_comparison_table(df_full, df_kfold)

    print("\n=== Comparison table ===")
    display_cols = [
        "majority", "guided_bertopic", "sup_svm",
        "episc_flash", "episc_pro",
    ]
    print(df[display_cols].round(3).to_string())

    # Figures
    print("\nGenerating figures...")
    plot_comparison(
        df,
        repo_root / "paper/figures/fig_baseline_comparison.pdf",
    )
    plot_corpus_audit(
        df,
        repo_root / "paper/figures/fig_corpus_audit.pdf",
    )

    # LaTeX table
    tex = build_latex_table(df)
    table_path = repo_root / "paper/tables/tab_baseline_comparison.tex"
    table_path.write_text(tex)
    print(f"  Saved: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
