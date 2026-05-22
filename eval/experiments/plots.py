"""Matplotlib helpers for the experiment reporting pipeline.

Produces the two figures referenced by §6 of the paper:

  - baseline_comparison.pdf : per-task Jaccard with one bar per baseline.
  - ablation_components.pdf : grouped bars for the metadata-only -> random
    chunks -> EpiScope progression. Only produced if the experiment's
    paper_role for at least three baselines is 'ablation'.

The figures use matplotlib's default backend and a print-safe colour set
(no seaborn dependency). All colours and labels are taken from the
experiment config so the same code generates the paper-main figure and
the ablation figure without per-experiment edits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: avoid GUI dependencies during eval runs.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification.experiments.config import ExperimentConfig


PAPER_ROLE_COLOURS = {
    "main": "#3366cc",
    "appendix": "#999999",
    "ablation": "#dc3912",
}


def _baseline_order(experiment: ExperimentConfig) -> list[str]:
    """Order baselines by their position in the config (paper-table order)."""
    return [spec.id for spec in experiment.baselines]


def _label_for(experiment: ExperimentConfig, baseline_id: str) -> str:
    return experiment.baseline_by_id(baseline_id).label


def _role_for(experiment: ExperimentConfig, baseline_id: str) -> str:
    return experiment.baseline_by_id(baseline_id).paper_role


def _task_panels(summary: pd.DataFrame, tasks: Sequence[str]) -> list[str]:
    """Return tasks for which at least one baseline produced a Jaccard mean."""
    return [
        task
        for task in tasks
        if not summary[summary["task"] == task]["jaccard_samples_mean"].dropna().empty
    ]


def emit_baseline_comparison(
    *,
    experiment: ExperimentConfig,
    summary: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Per-task Jaccard bar chart with one bar per baseline."""
    order = _baseline_order(experiment)
    tasks = _task_panels(summary, experiment.tasks)
    if not tasks:
        print(f"[WARN] no Jaccard values available; skipping {out_path.name}.")
        return out_path

    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.4 * len(tasks), 4.2), sharey=True
    )
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        rows = summary[summary["task"] == task].set_index("baseline_id")
        means = []
        stds = []
        labels = []
        colours = []
        for bid in order:
            if bid not in rows.index:
                continue
            row = rows.loc[bid]
            means.append(row.get("jaccard_samples_mean") or 0.0)
            stds.append(row.get("jaccard_samples_std") or 0.0)
            labels.append(_label_for(experiment, bid))
            colours.append(PAPER_ROLE_COLOURS.get(_role_for(experiment, bid), "#888"))
        positions = np.arange(len(means))
        ax.bar(positions, means, yerr=stds, color=colours, edgecolor="black", linewidth=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(task.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Jaccard ($J$)") if ax is axes[0] else ax.set_ylabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Experiment: {experiment.name}    "
        f"(baselines coloured by paper role)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_ablation_components(
    *,
    experiment: ExperimentConfig,
    summary: pd.DataFrame,
    out_path: Path,
) -> Path | None:
    """Grouped bars for ablation experiments (paper_role == 'ablation').

    Skipped (returns None) unless at least three baselines have
    paper_role == 'ablation', which is the design-space premise of §6.1.3.
    """
    ablation_specs = experiment.baselines_by_role("ablation")
    if len(ablation_specs) < 3:
        return None
    ids = [s.id for s in ablation_specs]
    tasks = _task_panels(summary[summary["baseline_id"].isin(ids)], experiment.tasks)
    if not tasks:
        return None

    n_bars = len(ids)
    bar_width = 0.8 / n_bars
    positions = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(2.5 + 1.2 * len(tasks), 4.2))
    for i, spec in enumerate(ablation_specs):
        means = []
        stds = []
        for task in tasks:
            row = summary[
                (summary["baseline_id"] == spec.id) & (summary["task"] == task)
            ]
            if row.empty:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(float(row["jaccard_samples_mean"].iloc[0] or 0.0))
                stds.append(float(row["jaccard_samples_std"].iloc[0] or 0.0))
        offsets = positions + (i - (n_bars - 1) / 2) * bar_width
        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=stds,
            label=spec.label,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([t.replace("_", " ") for t in tasks])
    ax.set_ylabel("Jaccard ($J$)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Component ablation: {experiment.name}")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_baseline_figures(
    *,
    experiment: ExperimentConfig,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """Emit the standard set of figures for an experiment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(
        emit_baseline_comparison(
            experiment=experiment,
            summary=summary,
            out_path=out_dir / "baseline_comparison.pdf",
        )
    )
    ablation_path = emit_ablation_components(
        experiment=experiment,
        summary=summary,
        out_path=out_dir / "ablation_components.pdf",
    )
    if ablation_path is not None:
        paths.append(ablation_path)
    return paths
