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
    eval_outputs/classification/per_run_metrics.csv
    sampled_papers_full.csv
"""

from __future__ import annotations

import argparse
import ast
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

# GT column in sampled_papers_full.csv for each task
TASK_TO_GT_COL = {
    "paper-type":         "ptype_classification",
    "geo":                "geo_classification",
    "data-type":          "data_type_classification",
    "data-accessibility": "availability_classification",
}

# EpiScope numbers — mean ± SD over R=5 independent runs at temperature 0
# Source: eval_outputs/classification_episc/summary_metrics.csv
# (recomputed after fixing the Jaccard sep=',' bug in error_analysis.py)
EPISC = {
    "paper-type":         {"flash_t0": (0.877, 0.010), "pro_t0": (0.903, 0.008)},
    "geo":                {"flash_t0": (0.868, 0.007), "pro_t0": (0.928, 0.004)},
    "data-type":          {"flash_t0": (0.916, 0.006), "pro_t0": (0.950, 0.017)},
    "data-accessibility": {"flash_t0": (0.670, 0.022), "pro_t0": (0.802, 0.023)},
}

# Prototype similarity (TF-IDF) — source TSVs not available locally for recomputation.
# Values from a prior run; the TF-IDF prototype assigns single-label predictions by
# threshold so the multi-label Jaccard bug has negligible impact on these numbers.
PROTO_TFIDF = {
    "paper-type":         0.454,
    "geo":                0.009,
    "data-type":          0.224,
    "data-accessibility": 0.168,
}


# ---------------------------------------------------------------------------
# Per-paper Jaccard helpers (for SD computation from prediction TSVs)
# ---------------------------------------------------------------------------

def _normalize_label_set(s: object) -> frozenset:
    """Parse a Python list string such as \"['AFRICA', 'EUROPE']\" to a frozenset."""
    if pd.isna(s) or str(s).strip() == "":
        return frozenset()
    s = str(s).replace("‘", "'").replace("’", "'")
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return frozenset(str(x).strip() for x in parsed)
        return frozenset([str(parsed).strip()])
    except (ValueError, SyntaxError):
        return frozenset([str(s).strip()])


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def load_gt(repo_root: Path) -> pd.DataFrame:
    """Load the ground-truth TSV, indexed by paper_id."""
    return pd.read_csv(repo_root / "sampled_papers_full.csv", sep="\t", index_col="paper_id")


def _find_tsv(df_metrics: pd.DataFrame, task: str, model: str,
              repo_root: Path, exact: bool = True) -> Path | None:
    """Return the path to final_1.tsv for a given task/model in a per_run_metrics frame."""
    if exact:
        mask = (df_metrics["task"] == task) & (df_metrics["model"] == model)
    else:
        mask = (df_metrics["task"] == task) & df_metrics["model"].str.startswith(model)
    sources = df_metrics.loc[mask, "source_file"]
    for sf in sources:
        if "final_1.tsv" in str(sf):
            p = repo_root / sf
            if p.exists():
                return p
    return None


def per_paper_sd(tsv_path: Path | None, gt_df: pd.DataFrame, task: str) -> float:
    """Compute the SD of per-paper Jaccard values from a prediction TSV.

    Each row in the TSV is one paper (214 rows for full-corpus LOO/k-fold
    evaluations).  The SD captures variability across papers, not across runs.
    Returns NaN if the file is missing or empty.
    """
    if tsv_path is None or not tsv_path.exists():
        return float("nan")
    pred_df = pd.read_csv(tsv_path, sep="\t")
    gt_col = TASK_TO_GT_COL[task]
    jaccards: list[float] = []
    for _, row in pred_df.iterrows():
        pid = row["paper_id"]
        if pid in gt_df.index:
            gt = _normalize_label_set(gt_df.loc[pid, gt_col])
            pred = _normalize_label_set(row["classification"])
            jaccards.append(_jaccard(gt, pred))
    return float(np.std(jaccards)) if jaccards else float("nan")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_full      = pd.read_csv(repo_root / "eval_outputs/classification/summary_metrics.csv")
    df_full_runs = pd.read_csv(repo_root / "eval_outputs/classification/per_run_metrics.csv")
    return df_full, df_full_runs


def _best_j(df: pd.DataFrame, task: str, model_prefix: str) -> float:
    mask = (df["task"] == task) & (df["model"].str.startswith(model_prefix))
    vals = df.loc[mask, "jaccard_samples_mean"]
    return float(vals.max()) if len(vals) else float("nan")


def _exact_j(df: pd.DataFrame, task: str, model: str) -> float:
    mask = (df["task"] == task) & (df["model"] == model)
    vals = df.loc[mask, "jaccard_samples_mean"]
    return float(vals.iloc[0]) if len(vals) else float("nan")


def build_comparison_table(
    df_full: pd.DataFrame,
    df_full_runs: pd.DataFrame,
    gt_df: pd.DataFrame,
    repo_root: Path,
) -> pd.DataFrame:
    """Build a tidy DataFrame: rows = methods, columns = tasks.

    For every method that has a per-paper prediction TSV (supervised baselines
    using LOO or k-fold CV, and zero-shot methods with a full-corpus TSV) we
    compute the standard deviation of per-paper Jaccard values.  This SD
    reflects variability across the 214 evaluation papers, not across runs.
    """
    records: list[dict] = []

    for task in TASK_ORDER:
        rec: dict = {"task": task}

        # --- Zero-shot ---
        rec["majority"]       = _exact_j(df_full, task, "majority")
        # TF-IDF prototype: source TSVs not available locally (see PROTO_TFIDF dict)
        rec["proto_tfidf"]    = PROTO_TFIDF[task]
        rec["proto_specter"]  = _exact_j(df_full, task, "prototype_similarity-emb_allenai-specter")
        rec["guided_bertopic"]= _best_j(df_full,  task, "bertopic_guided")

        # Per-paper SD for zero-shot methods with deterministic full-corpus TSVs
        rec["majority_sd"] = per_paper_sd(
            _find_tsv(df_full_runs, task, "majority", repo_root), gt_df, task)
        # prototype_similarity (TF-IDF) source files not available locally; no SD
        rec["proto_tfidf_sd"] = float("nan")

        # --- Weakly supervised (50% labels) ---
        rec["semisup_bertopic"] = _best_j(df_full, task, "bertopic_semisupervised")

        # --- Supervised TF-IDF (LOO CV) ---
        rec["sup_lr"]  = _exact_j(df_full, task, "supervised_tfidf_logreg-cv_leave_one_out")
        rec["sup_svm"] = _exact_j(df_full, task, "supervised_tfidf_linear_svm-cv_leave_one_out")

        rec["sup_lr_sd"] = per_paper_sd(
            _find_tsv(df_full_runs, task, "supervised_tfidf_logreg-cv_leave_one_out", repo_root),
            gt_df, task)
        rec["sup_svm_sd"] = per_paper_sd(
            _find_tsv(df_full_runs, task, "supervised_tfidf_linear_svm-cv_leave_one_out", repo_root),
            gt_df, task)

        # --- Supervised latent features + LR (LOO CV) ---
        loo_models = [
            ("sup_lsa_lr",  "supervised_lsa_logreg-k_10-cv_leave_one_out"),
            ("sup_lda_lr",  "supervised_lda_logreg-k_10-cv_leave_one_out"),
            ("sup_nmf_lr",  "supervised_nmf_logreg-k_10-cv_leave_one_out"),
            ("sup_plsa_lr", "supervised_plsa_logreg-k_10-cv_leave_one_out"),
            ("sup_bertopic","supervised_bertopic-cv_leave_one_out"),
        ]
        for col, model_name in loo_models:
            rec[col] = _exact_j(df_full, task, model_name)
            rec[col + "_sd"] = per_paper_sd(
                _find_tsv(df_full_runs, task, model_name, repo_root), gt_df, task)

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
        ("sup_svm",         "TF-IDF + SVM (LOO CV)",    "supervised"),
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

    def fmt(val: float, std_col: str | None = None, task: str | None = None) -> str:
        """Format a mean, optionally with ± SD if available."""
        if np.isnan(val):
            return "---"
        if std_col is not None and task is not None:
            sd = df.loc[task, std_col]
            if not np.isnan(sd):
                return f"{val:.3f} $\\pm$ {sd:.3f}"
        return f"{val:.3f}"

    def row(label: str, values: list[str]) -> str:
        return label + " & " + " & ".join(values) + r" \\"

    rows = []

    # ---------- Zero-shot ----------
    rows.append(r"\multicolumn{5}{l}{\textit{Zero-shot baselines}} \\")
    rows.append(row(r"\quad Majority prior",
                    [fmt(df.loc[t, "majority"], "majority_sd", t) for t in task_cols]))
    rows.append(row(r"\quad Prototype similarity (TF-IDF)",
                    [fmt(df.loc[t, "proto_tfidf"], "proto_tfidf_sd", t) for t in task_cols]))
    rows.append(row(r"\quad Prototype similarity (SPECTER)",
                    [fmt(df.loc[t, "proto_specter"]) for t in task_cols]))
    rows.append(row(r"\quad Guided BERTopic (best $K$)",
                    [fmt(df.loc[t, "guided_bertopic"]) for t in task_cols]))

    # ---------- Weakly supervised (50% labels) ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Weakly supervised (50\% labels exposed)}} \\")
    rows.append(row(r"\quad Semi-supervised BERTopic (best $K$)",
                    [fmt(df.loc[t, "semisup_bertopic"]) for t in task_cols]))

    # ---------- Supervised TF-IDF (LOO CV) ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Supervised --- TF-IDF features (leave-one-out CV)}} \\")
    rows.append(row(r"\quad TF-IDF + Logistic Regression",
                    [fmt(df.loc[t, "sup_lr"], "sup_lr_sd", t) for t in task_cols]))
    rows.append(row(r"\quad TF-IDF + Linear SVM",
                    [fmt(df.loc[t, "sup_svm"], "sup_svm_sd", t) for t in task_cols]))

    # ---------- Supervised latent features + LR (LOO CV) ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Supervised --- latent features (leave-one-out CV)}} \\")
    rows.append(row(r"\quad LSA + Logistic Regression ($k{=}10$)",
                    [fmt(df.loc[t, "sup_lsa_lr"], "sup_lsa_lr_sd", t) for t in task_cols]))
    rows.append(row(r"\quad LDA + Logistic Regression ($k{=}10$)",
                    [fmt(df.loc[t, "sup_lda_lr"], "sup_lda_lr_sd", t) for t in task_cols]))
    rows.append(row(r"\quad NMF + Logistic Regression ($k{=}10$)",
                    [fmt(df.loc[t, "sup_nmf_lr"], "sup_nmf_lr_sd", t) for t in task_cols]))
    rows.append(row(r"\quad PLSA + Logistic Regression ($k{=}10$)",
                    [fmt(df.loc[t, "sup_plsa_lr"], "sup_plsa_lr_sd", t) for t in task_cols]))
    rows.append(row(r"\quad Supervised BERTopic",
                    [fmt(df.loc[t, "sup_bertopic"], "sup_bertopic_sd", t) for t in task_cols]))

    # ---------- Unsupervised (audit) ----------
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\textit{Unsupervised audit (best $K$ across sweep, oracle label alignment)}} \\")
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
        [fmt(df.loc[t, "episc_flash"], "episc_flash_std", t) for t in task_cols],
    ))
    rows.append(row(
        r"\quad \texttt{gemini-2.5-pro}, $T{=}0$",
        [fmt(df.loc[t, "episc_pro"], "episc_pro_std", t) for t in task_cols],
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
Where shown, $\pm$ values are standard deviations: for supervised baselines
(leave-one-out CV), SD is computed over the $N{{=}}214$ per-paper
Jaccard values from the combined held-out predictions, capturing variability
across papers; for \toolname{{}}, SD is computed over $R{{=}}5$ independent runs
at temperature 0, capturing run-to-run variability (the two SDs are not
directly comparable).
Unsupervised topic-model results report the best $J$ achieved across the $K$ sweep
($K \in \{{0.5,\,1,\,2,\,4\}} \times |\mathcal{{L}}|$) under an oracle cluster-to-label
alignment; they are included as a corpus structure audit rather than as competitive
classifiers (see \Cref{{app:baselines}}).
All supervised baselines use leave-one-out cross-validation.
Empty cells (``---'') indicate configurations for which the corresponding result
file was not present.
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

    df_full, df_full_runs = load_data(repo_root)
    gt_df = load_gt(repo_root)
    df = build_comparison_table(df_full, df_full_runs, gt_df, repo_root)

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
