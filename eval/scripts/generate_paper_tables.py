"""
Generate LaTeX table files for the EpiScope paper.

Usage
-----
    python eval/scripts/generate_paper_tables.py \\
        --eval-dir eval_outputs/classification_episc \\
        --out-dir paper/tables

The --eval-dir must contain:
    summary_metrics.csv
    within_config_consistency.csv
    containment_pro_vs_flash.csv
    per_label_summary.csv          (can live in --per-label-dir instead)

Each file corresponds to one or more LaTeX tables written to --out-dir.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Config: canonical display names
# ---------------------------------------------------------------------------

# The four model/temperature configs in fixed display order
CONFIGS = [
    ("gemini-2-5-pro", "0.0"),
    ("gemini-2-5-pro", "1.0"),
    ("gemini-2-5-flash", "0.0"),
    ("gemini-2-5-flash", "1.0"),
]

# Canonical display name for tasks (ordered for table rows)
TASK_DISPLAY = {
    "geo": "Geography",
    "data-accessibility": "Data accessibility",
    "data-type": "Data type",
    "paper-type": "Paper type",
}

# Order for the 4 summary tables (tab_metrics_summary, tab_stability, app_tab_all_metrics)
TASK_ORDER = ["geo", "data-accessibility", "data-type", "paper-type"]

# For per-label and combined PRF tables
TASK_DISPLAY_TITLE = {
    "geo": "Geography",
    "data-type": "Data Type",
    "data-accessibility": "Data Accessibility",
    "paper-type": "Paper Type",
}

# task -> short name used in tex file names
TASK_TEX_KEY = {
    "geo": "geo",
    "data-accessibility": "access",
    "data-type": "datatype",
    "paper-type": "papertype",
}

# task -> caption note for per-label tables
TASK_PERLABEL_CAPTION = {
    "geo": "Per-label F\\textsubscript{1} for geography.",
    "data-accessibility": "Per-label F\\textsubscript{1} for data accessibility.",
    "data-type": "Per-label F\\textsubscript{1} for data type.",
    "paper-type": (
        "Per-label F\\textsubscript{1} for paper type.\n"
        "Paper type is a single-label task."
    ),
}

# task -> app tab label
TASK_PERLABEL_LABEL = {
    "geo": "app:tab:lbl_geo_f1",
    "data-accessibility": "app:tab:lbl_access_f1",
    "data-type": "app:tab:lbl_datatype_f1",
    "paper-type": "app:tab:lbl_ptype_f1",
}

# Containment table task display (ordered by how they appear in the existing tables)
CONTAINMENT_TASK_ORDER = [
    ("geo", "Geography"),
    ("data-type", "Data type"),
    ("data-accessibility", "Data accessibility"),
    ("paper-type", "Paper type"),
]

# app_tab_all_metrics: row definitions
ALL_METRICS_ROWS = [
    ("jaccard_samples", "Jaccard"),
    ("micro_precision", "Micro precision"),
    ("micro_recall", "Micro recall"),
    ("micro_f1", r"Micro F$_1$"),
    ("macro_precision", "Macro precision"),
    ("macro_recall", "Macro recall"),
    ("macro_f1", r"Macro F$_1$"),
    ("subset_accuracy", "Subset accuracy"),
]

# minmax tables
MINMAX_TASK_INFO = {
    "geo": ("geography", "tab:minmax_geo"),
    "data-accessibility": ("data accessibility", "tab:minmax_access"),
    "data-type": ("data type", "tab:minmax_datatype"),
    "paper-type": ("paper type", "tab:minmax_papertype"),
}

MINMAX_ROWS = [
    ("jaccard_samples", r"$J_{\text{samples}}$"),
    ("micro_precision", r"$\text{Prec}_{\mu}$"),
    ("micro_recall", r"$\text{Rec}_{\mu}$"),
    ("micro_f1", r"$\text{F1}_{\mu}$"),
    ("macro_precision", r"$\text{Prec}_{\text{macro}}$"),
    ("macro_recall", r"$\text{Rec}_{\text{macro}}$"),
    ("macro_f1", r"$\text{F1}_{\text{macro}}$"),
    ("subset_accuracy", "EMR"),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sig_figs(value: float, n: int) -> int:
    """Return the number of decimal places needed for *n* significant figures."""
    if value == 0.0:
        return n  # arbitrary fallback
    magnitude = math.floor(math.log10(abs(value)))
    decimals = n - 1 - magnitude
    return max(0, decimals)


def _fmt_mean_std(mean: float, std: float, n_sig: int = 2) -> str:
    """
    Format a mean ± std pair as a LaTeX math string.

    The precision is determined by the standard deviation (n_sig sig figs on
    std), then the mean is rounded to the same number of decimal places.
    Edge case: std == 0 → show mean with 3 decimal places and '\\pm 0'.
    """
    if std == 0.0:
        return f"${mean:.3f}\\pm 0$"
    decimals = _sig_figs(std, n_sig)
    mean_r = round(mean, decimals)
    std_r = round(std, decimals)
    fmt = f"{{:.{decimals}f}}"
    return f"${fmt.format(mean_r)}\\pm {fmt.format(std_r)}$"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a percentage value."""
    return f"{value:.{decimals}f}"


def _lookup(df: pd.DataFrame, task: str, model: str, temperature: str, col: str) -> float:
    """Look up a single scalar from a summary DataFrame."""
    mask = (
        (df["task"] == task)
        & (df["model"] == model)
        & (df["temperature"] == temperature)
    )
    rows = df[mask]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][col])


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _header_two_models() -> str:
    """Common two-model column header block."""
    return (
        "& \\multicolumn{2}{c}{\\textbf{gemini-2.5-pro}}"
        " & \\multicolumn{2}{c}{\\textbf{gemini-2.5-flash}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "\\textbf{Task} & $T{=}0.0$ & $T{=}1.0$ & $T{=}0.0$ & $T{=}1.0$ \\\\"
    )


def gen_tab_metrics_summary(summary: pd.DataFrame) -> str:
    """tab_metrics_summary.tex — Jaccard mean±std for 4 tasks × 4 configs."""
    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(_header_two_models())
    lines.append(r"\midrule")

    for task in TASK_ORDER:
        cells = []
        for model, temp in CONFIGS:
            mean = _lookup(summary, task, model, temp, "jaccard_samples_mean")
            std = _lookup(summary, task, model, temp, "jaccard_samples_std")
            cells.append(_fmt_mean_std(mean, std))
        display = TASK_DISPLAY[task]
        # Pad task name to column width (match existing style)
        lines.append(f"{display:<19}& {cells[0]:<18} & {cells[1]:<18} & {cells[2]:<18} & {cells[3]:<18} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{")
    lines.append(
        r"Mean $\pm$ standard deviation of sample-averaged Jaccard similarity"
        r" across $R{=}5$ repeated runs per configuration."
    )
    lines.append(r"Paper type is a single-label task.")
    lines.append(r"The remaining tasks are multi-label.")
    lines.append(r"}")
    lines.append(r"\label{tab:metrics_summary}")
    lines.append(r"\label{tab:jaccard_all_configs}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def gen_tab_stability_all_configs(consistency: pd.DataFrame) -> str:
    """tab_stability_all_configs.tex — pct_stable for 4 tasks × 4 configs."""
    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(_header_two_models())
    lines.append(r"\midrule")

    for task in TASK_ORDER:
        cells = []
        for model, temp in CONFIGS:
            mask = (
                (consistency["task"] == task)
                & (consistency["model"] == model)
                & (consistency["temperature"] == temp)
            )
            rows = consistency[mask]
            if rows.empty:
                cells.append("---")
            else:
                cells.append(_fmt_pct(float(rows.iloc[0]["pct_stable"])))
        display = TASK_DISPLAY[task]
        lines.append(
            f"{display:<19}& {cells[0]:<6} & {cells[1]:<6} & {cells[2]:<6} & {cells[3]:<6} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{")
    lines.append(
        r"Percentage of papers whose predicted label set was identical across"
        r" five repeated runs under each configuration."
    )
    lines.append(
        r"A prediction is considered stable if the full label set is unchanged"
        r" across runs."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:stability_all_configs}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def gen_app_tab_all_metrics_all_configs(summary: pd.DataFrame) -> str:
    """app_tab_all_metrics_all_configs.tex — longtable, 4 task blocks."""
    header_row = (
        "& \\multicolumn{2}{c}{\\textbf{gemini-2.5-pro}}"
        " & \\multicolumn{2}{c}{\\textbf{gemini-2.5-flash}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "\\textbf{Metric} & $T{=}0.0$ & $T{=}1.0$ & $T{=}0.0$ & $T{=}1.0$ \\\\"
    )

    lines: list[str] = []
    lines.append(r"\begin{longtable}{lcccc}")
    lines.append(r"\caption{")
    lines.append(
        r"Aggregate agreement metrics by task and configuration on the fixed"
        r" evaluation set (214 papers)."
    )
    lines.append(r"Values are mean $\pm$ SD across $R{=}5$ repeated runs.")
    lines.append(r"}\label{app:tab:all_metrics_all_configs}\\")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{\small\itshape (continued on next page)}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    first_task = True
    for task in TASK_ORDER:
        if first_task:
            lines.append(r"")
            lines.append(r"\addlinespace[2pt]")
            first_task = False
        else:
            lines.append(r"")
            lines.append(r"\addlinespace[4pt]")
        display = TASK_DISPLAY[task]
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{display}}}}}\\\\")
        lines.append(r"\addlinespace[2pt]")

        for col_key, row_label in ALL_METRICS_ROWS:
            cells = []
            for model, temp in CONFIGS:
                mean = _lookup(summary, task, model, temp, f"{col_key}_mean")
                std = _lookup(summary, task, model, temp, f"{col_key}_std")
                cells.append(_fmt_mean_std(mean, std))
            lines.append(
                f"{row_label:<17}& {cells[0]:<18} & {cells[1]:<18}"
                f" & {cells[2]:<18} & {cells[3]:<18} \\\\"
            )

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _containment_table(containment: pd.DataFrame, temperature: str) -> str:
    """Generate one containment table for a given temperature."""
    subset = containment[containment["temperature"] == temperature].copy()

    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lrrrcc}")
    lines.append(r"\toprule")
    lines.append(
        r"Task & $|E_{\text{pro}}|$ & $|E_{\text{flash}}|$ & $|E_{\cap}|$ &"
    )
    lines.append(r"pro$\subseteq$flash & flash$\subseteq$pro \\")
    lines.append(r"\midrule")

    for task, display in CONTAINMENT_TASK_ORDER:
        row = subset[subset["task"] == task]
        if row.empty:
            n_pro = n_flash = n_shared = "---"
            pro_in = flash_in = "---"
        else:
            r = row.iloc[0]
            n_pro = int(r["n_pro_errors"])
            n_flash = int(r["n_flash_errors"])
            n_shared = int(r["n_shared"])
            pro_pct = r["pro_in_flash_pct"]
            flash_pct = r["flash_in_pro_pct"]
            pro_in = f"{pro_pct:.1f}\\%" if pro_pct is not None and not math.isnan(float(pro_pct)) else "---"
            flash_in = f"{flash_pct:.1f}\\%" if flash_pct is not None and not math.isnan(float(flash_pct)) else "---"

        lines.append(
            f"{display:<19}& {n_pro!s:<4} & {n_flash!s:<4} & {n_shared!s:<4}"
            f" & {pro_in:<8} & {flash_in:<8} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    temp_display = temperature
    if temperature == "0.0":
        label_key = "tab:containment_t0"
        caption = (
            r"\textbf{Containment of ever-wrong sets at $T{=}0.0$.}" + "\n"
            r"For each task, $E$ is the set of papers that are incorrectly"
            r" labeled in at least one run." + "\n"
            r"$|E_{\text{pro}}|$ and $|E_{\text{flash}}|$ report ever-wrong"
            r" set sizes for the two backends." + "\n"
            r"$|E_{\cap}|$ reports overlap."
        )
    else:
        label_key = "tab:containment_t1"
        caption = (
            r"\textbf{Containment of ever-wrong sets at $T{=}1.0$.}" + "\n"
            r"Definitions as in \Cref{tab:containment_t0}."
        )

    lines.append(r"\caption{")
    lines.append(caption)
    lines.append(r"}")
    lines.append(f"\\label{{{label_key}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def gen_tab_containment_t0(containment: pd.DataFrame) -> str:
    return _containment_table(containment, "0.0")


def gen_tab_containment_t1(containment: pd.DataFrame) -> str:
    return _containment_table(containment, "1.0")


def _minmax_table(summary: pd.DataFrame, task: str) -> str:
    """Generate one dispersion summary table for gemini-2-5-pro T=0."""
    model = "gemini-2-5-pro"
    temperature = "0.0"

    task_name, label_key = MINMAX_TASK_INFO[task]
    mask = (
        (summary["task"] == task)
        & (summary["model"] == model)
        & (summary["temperature"] == temperature)
    )
    row = summary[mask]
    if row.empty:
        row_vals = {}
    else:
        row_vals = row.iloc[0].to_dict()

    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{\\textbf{{Dispersion summary for {task_name}"
        r" (gemini-2.5-pro, $T=0$).}"   # closes \textbf
        r" Min--max ranges are computed across $R{=}5$ repeated runs on the"
        r" fixed evaluation set (214 papers).}"  # closes \caption
    )
    lines.append(f"\\label{{{label_key}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Mean $\pm$ SD & Min & Max \\")
    lines.append(r"\midrule")

    for col_key, row_label in MINMAX_ROWS:
        mean = float(row_vals.get(f"{col_key}_mean", float("nan")))
        std = float(row_vals.get(f"{col_key}_std", float("nan")))
        mn = float(row_vals.get(f"{col_key}_min", float("nan")))
        mx = float(row_vals.get(f"{col_key}_max", float("nan")))
        # mean±std: use raw values (no trailing-zero rounding) matching existing format
        if math.isnan(mean):
            cell_ms = "---"
            cell_min = "---"
            cell_max = "---"
        else:
            # Format: fixed 4 decimal places (matching existing tables)
            cell_ms = f"${mean:.4f}\\pm {std:.4f}$"
            cell_min = f"{mn:.4f}"
            cell_max = f"{mx:.4f}"
        lines.append(
            f"{row_label:<29}& {cell_ms} & {cell_min} & {cell_max} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def gen_tab_minmax(summary: pd.DataFrame, task: str) -> str:
    return _minmax_table(summary, task)


def _app_tab_lbl_f1(per_label_summary: pd.DataFrame, task: str) -> str:
    """Generate one per-label F1 table for a task."""
    subset = per_label_summary[per_label_summary["task"] == task].copy()

    # Get labels ordered by support descending (stable sort by name within same support)
    if subset.empty:
        labels_ordered = []
    else:
        label_support = (
            subset[["label", "support"]]
            .drop_duplicates("label")
            .sort_values(["support", "label"], ascending=[False, True])
        )
        labels_ordered = label_support["label"].tolist()

    caption = TASK_PERLABEL_CAPTION[task]
    label_key = TASK_PERLABEL_LABEL[task]

    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{\textbf{gemini-2.5-pro}}"
        r" & \multicolumn{2}{c}{\textbf{gemini-2.5-flash}} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(
        r"\textbf{Label (support)} & $T{=}0.0$ & $T{=}1.0$ & $T{=}0.0$ & $T{=}1.0$ \\"
    )
    lines.append(r"\midrule")

    for label in labels_ordered:
        label_rows = subset[subset["label"] == label]
        support = int(label_rows["support"].iloc[0]) if not label_rows.empty else 0

        cells = []
        for model, temp in CONFIGS:
            r = label_rows[
                (label_rows["model"] == model)
                & (label_rows["temperature"] == temp)
            ]
            if r.empty:
                cells.append("---")
            else:
                mean = float(r.iloc[0]["f1_mean"])
                std = float(r.iloc[0]["f1_std"])
                cells.append(_fmt_mean_std(mean, std).replace(" ", ""))
                # per-label F1 uses compact format like existing: no space around ±
                # Actually existing format uses no space: $0.971\pm 0.004$
                # Let's keep the space variant consistent with gen_tab_metrics_summary
                cells[-1] = _fmt_mean_std(mean, std)

        label_display = f"{label} ({support})"
        lines.append(
            f"{label_display:<29}& {cells[0]:<16} & {cells[1]:<16}"
            f" & {cells[2]:<16} & {cells[3]:<16} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{")
    lines.append(caption)
    lines.append(r"Values are mean $\pm$ SD across repeated runs.")
    lines.append(r"Numbers in parentheses denote label support in the evaluation set.")
    lines.append(r"}")
    lines.append(f"\\label{{{label_key}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def gen_app_tab_lbl_f1(per_label_summary: pd.DataFrame, task: str) -> str:
    return _app_tab_lbl_f1(per_label_summary, task)


# Task display names for combined PRF table (block headers)
COMBINED_TASK_ORDER = ["geo", "data-type", "data-accessibility", "paper-type"]
COMBINED_TASK_BLOCK_DISPLAY = {
    "geo": "Geography",
    "data-type": "Data Type",
    "data-accessibility": "Data Accessibility",
    "paper-type": "Paper Type",
}


def gen_combined_prf_table(per_label_summary: pd.DataFrame) -> str:
    """
    combined_prf_table.tex — single-config (gemini-2-5-pro, T=0.0) PRF longtable.
    Ordered: Geography, Data Type, Data Accessibility, Paper Type.
    Within each task, rows ordered by support descending.
    """
    model = "gemini-2-5-pro"
    temperature = "0.0"

    lines: list[str] = []
    lines.append(r"\begin{longtable}{llrrrr}")
    lines.append(
        r"\caption{Per-label precision, recall and F\textsubscript{1} for all"
        r" tasks (model: gemini-2-5-pro, $T=0.0$)."
    )
    lines.append(r"Within each task block rows are ordered by support (\#).")
    lines.append(r"Values are mean\,$\pm$\,SD across runs.}")
    lines.append(r"\label{app:tab:combined_prf}")
    lines.append(r"\label{tab:gemini-2-5-pro_T0.0:combined_prf} \\")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Task} & \textbf{Label} & \textbf{\#}"
        r" & \textbf{P} & \textbf{R} & \textbf{F\textsubscript{1}} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Task} & \textbf{Label} & \textbf{\#}"
        r" & \textbf{P} & \textbf{R} & \textbf{F\textsubscript{1}} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\small\itshape (continued on next page)} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for task_idx, task in enumerate(COMBINED_TASK_ORDER):
        task_data = per_label_summary[
            (per_label_summary["task"] == task)
            & (per_label_summary["model"] == model)
            & (per_label_summary["temperature"] == temperature)
        ].copy()

        if task_data.empty:
            labels_ordered = []
        else:
            labels_ordered = (
                task_data.sort_values(["support", "label"], ascending=[False, True])
                ["label"]
                .tolist()
            )

        n_labels = len(labels_ordered)
        block_display = COMBINED_TASK_BLOCK_DISPLAY[task]

        if task_idx > 0:
            lines.append(r"\midrule")

        for i, label in enumerate(labels_ordered):
            row = task_data[task_data["label"] == label]
            if row.empty:
                p_str = r_str = f_str = "---"
                support = "---"
            else:
                r0 = row.iloc[0]
                support = int(r0["support"])
                p_str = _fmt_mean_std(float(r0["precision_mean"]), float(r0["precision_std"]))
                r_str = _fmt_mean_std(float(r0["recall_mean"]), float(r0["recall_std"]))
                f_str = _fmt_mean_std(float(r0["f1_mean"]), float(r0["f1_std"]))

                # Combined PRF uses compact format without spaces: value$\pm$value
                # Matching existing file: 0.944$\pm$0.007
                # We use the helper but replace the spaces-in-formula style
                def _compact(mean: float, std: float) -> str:
                    if std == 0.0:
                        return f"{mean:.3f}$\\pm${0}"
                    decimals = _sig_figs(std, 2)
                    mean_r = round(mean, decimals)
                    std_r = round(std, decimals)
                    fmt = f"{{:.{decimals}f}}"
                    return f"{fmt.format(mean_r)}$\\pm${fmt.format(std_r)}"

                p_str = _compact(float(r0["precision_mean"]), float(r0["precision_std"]))
                r_str = _compact(float(r0["recall_mean"]), float(r0["recall_std"]))
                f_str = _compact(float(r0["f1_mean"]), float(r0["f1_std"]))

            if i == 0:
                task_col = f"\\multirow{{{n_labels}}}{{*}}{{\\textbf{{{block_display}}}}}"
            else:
                task_col = " "

            lines.append(
                f"{task_col} & {label} & {support}"
                f" & {p_str} & {r_str} & {f_str} \\\\"
            )

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table files for the EpiScope paper."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help=(
            "Directory containing summary_metrics.csv, "
            "within_config_consistency.csv, containment_pro_vs_flash.csv."
        ),
    )
    parser.add_argument(
        "--per-label-dir",
        default=None,
        help=(
            "Directory containing per_label_summary.csv. "
            "Defaults to --eval-dir if not given."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="paper/tables",
        help="Directory where .tex files will be written.",
    )
    return parser


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype={"temperature": str})


def main() -> None:
    args = build_parser().parse_args()
    eval_dir = Path(args.eval_dir)
    per_label_dir = Path(args.per_label_dir) if args.per_label_dir else eval_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    summary = _load_csv(eval_dir / "summary_metrics.csv")
    consistency = _load_csv(eval_dir / "within_config_consistency.csv")
    containment = _load_csv(eval_dir / "containment_pro_vs_flash.csv")
    per_label_summary = _load_csv(per_label_dir / "per_label_summary.csv")

    # Normalise temperature column to string with one decimal
    for df in (summary, consistency, containment, per_label_summary):
        if "temperature" in df.columns:
            df["temperature"] = df["temperature"].astype(str)

    def write(filename: str, content: str) -> None:
        path = out_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  Wrote {path}")

    print(f"Writing tables to {out_dir}/ ...")

    write("tab_metrics_summary.tex", gen_tab_metrics_summary(summary))
    write("tab_stability_all_configs.tex", gen_tab_stability_all_configs(consistency))
    write(
        "app_tab_all_metrics_all_configs.tex",
        gen_app_tab_all_metrics_all_configs(summary),
    )
    write("tab_containment_t0.tex", gen_tab_containment_t0(containment))
    write("tab_containment_t1.tex", gen_tab_containment_t1(containment))

    for task, tex_key in TASK_TEX_KEY.items():
        write(f"tab_minmax_{tex_key}.tex", gen_tab_minmax(summary, task))
        write(
            f"app_tab_lbl_{tex_key}_f1.tex",
            gen_app_tab_lbl_f1(per_label_summary, task),
        )

    write("combined_prf_table.tex", gen_combined_prf_table(per_label_summary))

    print("Done.")


if __name__ == "__main__":
    main()
