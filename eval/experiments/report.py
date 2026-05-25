"""Evaluate an experiment's baselines and emit paper-ready artefacts.

Reads the same JSON experiment config that ``classification.experiments.run``
consumed, plus the manifest the runner wrote to ``<eval_root>/manifest.json``.
For each baseline whose run_root contains result TSVs, computes per-run
metrics (Jaccard, F1, etc.), aggregates them per (baseline, task), runs the
significance pairs declared in the config, and writes:

  <eval_root>/per_run_metrics.csv         # one row per result file
  <eval_root>/per_baseline_summary.csv    # one row per (baseline, task)
  <eval_root>/significance.csv            # bootstrap + McNemar per pair
  <eval_root>/summary.json                # machine-readable rollup
  <eval_root>/figures/                    # matplotlib outputs (see plots.py)

Usage
-----

    python -m eval.experiments.report \\
        --config eval/configs/classification/paper_main.json

The script is idempotent: re-running with the same config rewrites the
artefacts in place. The manifest can be passed explicitly with --manifest
to evaluate against a manifest that was moved or renamed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd

from classification.experiments.config import (
    ExperimentConfig,
    load_experiment_config,
)
from eval.classification.error_analysis import (
    load_merged_predictions,
    per_run_metrics,
)
from eval.classification.significance import pairwise_significance
from eval.classification.summary import METRIC_COLUMNS
from eval.common.io import write_csv


def _resolve_run_root(spec_run_root: str) -> list[Path]:
    """Resolve a manifest run_root entry to one or more absolute paths.

    Globs (``*``, ``?``) are expanded against the repository root. Plain
    paths are returned as a single-element list. Non-existent entries
    produce an empty list (the caller warns).
    """
    raw = str(spec_run_root).strip()
    if not raw:
        return []
    if any(ch in raw for ch in "*?["):
        # glob-relative paths anchor at ROOT; absolute globs anchor at "/".
        if raw.startswith("/"):
            return sorted(p for p in Path("/").glob(raw.lstrip("/")) if p.is_dir())
        return sorted(p for p in ROOT.glob(raw) if p.is_dir())
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return [path] if path.exists() else []


def _baseline_run_roots(manifest: dict[str, Any]) -> list[tuple[str, str, list[Path]]]:
    """Return (baseline_id, baseline_label, [run_root, ...]) for each baseline.

    Each baseline can resolve to multiple run roots, either because the
    manifest already stored a list (multi-path ``external_run_root``) or
    because a single entry was a glob expanded at resolution time.
    """
    rows: list[tuple[str, str, list[Path]]] = []
    for entry in manifest.get("baselines", []):
        raw = entry.get("run_root")
        if not raw:
            continue
        candidates = raw if isinstance(raw, list) else [raw]
        paths: list[Path] = []
        for candidate in candidates:
            paths.extend(_resolve_run_root(candidate))
        if not paths:
            print(
                f"[WARN] baseline {entry['id']!r}: no existing paths under "
                f"run_root={raw!r}; skipping."
            )
            continue
        rows.append((entry["id"], entry.get("label", entry["id"]), paths))
    return rows


def compute_per_run_metrics(
    *,
    experiment: ExperimentConfig,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """One row per result TSV across all baselines in the manifest."""
    frames: list[pd.DataFrame] = []
    gt = experiment.ground_truth_csv
    for baseline_id, label, run_roots in _baseline_run_roots(manifest):
        df = per_run_metrics(
            ground_truth_path=gt,
            run_roots=[str(p) for p in run_roots],
            pattern="final_*.tsv",
        )
        if df.empty:
            print(
                f"[WARN] baseline {baseline_id!r}: no result TSVs found under "
                f"{[str(p) for p in run_roots]}."
            )
            continue
        df.insert(0, "baseline_id", baseline_id)
        df.insert(1, "baseline_label", label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_per_baseline(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per_run_metrics by (baseline_id, task) with mean/std/n."""
    if per_run.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group = per_run.groupby(["baseline_id", "baseline_label", "task"], as_index=False)
    for (bid, label, task), frame in group:
        row: dict[str, Any] = {
            "baseline_id": bid,
            "baseline_label": label,
            "task": task,
            "n_runs": len(frame),
            "n_predictions_mean": float(frame["n_predictions"].mean()),
            "n_skipped_mean": float(frame["n_skipped"].mean()),
        }
        for col in METRIC_COLUMNS:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(values) == 0:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
                row[f"{col}_min"] = None
                row[f"{col}_max"] = None
                continue
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "baseline_id"]).reset_index(drop=True)


def _merged_predictions_for_significance(
    *,
    experiment: ExperimentConfig,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Build a merged predictions DataFrame whose 'method' column is baseline_id.

    The base ``load_merged_predictions`` keys methods by ``model`` derived
    from the path. For experiment-level significance we want to compare
    baselines by their config id, so we relabel before stitching.
    """
    frames: list[pd.DataFrame] = []
    gt = experiment.ground_truth_csv
    for baseline_id, label, run_roots in _baseline_run_roots(manifest):
        merged = load_merged_predictions(
            ground_truth_path=gt,
            run_roots=[str(p) for p in run_roots],
            pattern="final_*.tsv",
        )
        if merged.empty:
            continue
        merged = merged.copy()
        merged["baseline_id"] = baseline_id
        merged["baseline_label"] = label
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_significance(
    *,
    experiment: ExperimentConfig,
    manifest: dict[str, Any],
    bootstrap_iterations: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if not experiment.significance_pairs:
        return pd.DataFrame()
    merged = _merged_predictions_for_significance(
        experiment=experiment, manifest=manifest
    )
    if merged.empty:
        print("[WARN] No predictions found for any baseline; skipping significance.")
        return pd.DataFrame()
    pairs = [(p.a, p.b) for p in experiment.significance_pairs]
    return pairwise_significance(
        merged,
        pairs=pairs,
        method_cols=("baseline_id",),
        B=bootstrap_iterations,
        alpha=alpha,
        random_state=experiment.random_state,
    )


def write_summary_json(
    *,
    experiment: ExperimentConfig,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    out_path: Path,
) -> None:
    """Compact JSON rollup for the paper's figure/table generators."""
    rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            rows.append(
                {
                    "baseline_id": row["baseline_id"],
                    "baseline_label": row["baseline_label"],
                    "task": row["task"],
                    "n_runs": int(row["n_runs"]),
                    "jaccard_mean": row.get("jaccard_samples_mean"),
                    "jaccard_std": row.get("jaccard_samples_std"),
                    "micro_f1_mean": row.get("micro_f1_mean"),
                    "subset_accuracy_mean": row.get("subset_accuracy_mean"),
                }
            )
    payload = {
        "experiment_name": experiment.name,
        "ground_truth_csv": experiment.ground_truth_csv,
        "tasks": list(experiment.tasks),
        "n_baselines_evaluated": int(summary["baseline_id"].nunique()) if not summary.empty else 0,
        "summary": rows,
        "significance": (
            significance.to_dict(orient="records") if not significance.empty else []
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_report(
    config_path: str | Path,
    *,
    manifest_path: Optional[str | Path] = None,
    bootstrap_iterations: int = 10_000,
    alpha: float = 0.05,
    no_plots: bool = False,
) -> Path:
    experiment = load_experiment_config(config_path)
    eval_root = Path(experiment.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else eval_root / "manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run "
            f"`python -m classification.experiments.run --config {config_path}` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"=== reporting experiment {experiment.name!r} ===")
    print(f"    manifest: {manifest_path}")
    print(f"    eval_root: {eval_root}")

    per_run = compute_per_run_metrics(experiment=experiment, manifest=manifest)
    if per_run.empty:
        print("[WARN] No per-run metrics were computed; nothing to report.")
    else:
        write_csv(per_run, eval_root / "per_run_metrics.csv")
        print(f"    wrote {eval_root / 'per_run_metrics.csv'}  ({len(per_run)} rows)")

    summary = summarize_per_baseline(per_run)
    if not summary.empty:
        write_csv(summary, eval_root / "per_baseline_summary.csv")
        print(f"    wrote {eval_root / 'per_baseline_summary.csv'}  ({len(summary)} rows)")

    significance = compute_significance(
        experiment=experiment,
        manifest=manifest,
        bootstrap_iterations=bootstrap_iterations,
        alpha=alpha,
    )
    if not significance.empty:
        write_csv(significance, eval_root / "significance.csv")
        print(f"    wrote {eval_root / 'significance.csv'}  ({len(significance)} rows)")

    write_summary_json(
        experiment=experiment,
        summary=summary,
        significance=significance,
        out_path=eval_root / "summary.json",
    )
    print(f"    wrote {eval_root / 'summary.json'}")

    if not no_plots and not summary.empty:
        from eval.experiments.plots import emit_baseline_figures

        figures_dir = eval_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        emit_baseline_figures(
            experiment=experiment,
            summary=summary,
            significance=significance,
            out_dir=figures_dir,
        )
        print(f"    wrote figures under {figures_dir}")

    return eval_root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the baselines listed in an experiment config and emit paper artefacts.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to the manifest written by classification.experiments.run. "
        "Defaults to <eval_root>/manifest.json.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run_report(
        args.config,
        manifest_path=args.manifest,
        bootstrap_iterations=args.bootstrap_iterations,
        alpha=args.alpha,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
