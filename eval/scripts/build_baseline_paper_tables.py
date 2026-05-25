"""Turn `per_run_metrics.csv` into compact, paper-ready pivot tables.

Consumes the CSV emitted by ``eval.scripts.run_classification_eval`` and writes
one wide pivot per chosen metric (rows = baselines, columns = tasks, cells =
``mean ± std (n)``), plus a single Markdown digest for quick inspection.

Usage
-----

    python -m eval.scripts.build_baseline_paper_tables \\
        --eval-dir eval_outputs/classification_baselines

The defaults emit Jaccard, micro-F1, macro-F1, and exact-match (subset
accuracy) pivots — the four numbers the paper actually references when
positioning baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.classification.paper_view import build_pivot, render_markdown


DEFAULT_METRICS: tuple[tuple[str, str], ...] = (
    ("jaccard_samples", "Jaccard (samples)"),
    ("micro_f1", "Micro F1"),
    ("macro_f1", "Macro F1"),
    ("subset_accuracy", "Exact match (subset accuracy)"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper-ready baseline pivot tables from per_run_metrics.csv.",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing per_run_metrics.csv (output of run_classification_eval).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write outputs. Defaults to --eval-dir.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help=(
            "Metric column to pivot on (repeatable). Defaults to: "
            "jaccard_samples, micro_f1, macro_f1, subset_accuracy."
        ),
    )
    return parser


def _run(eval_dir: Path, out_dir: Path, metrics: Sequence[tuple[str, str]]) -> None:
    per_run_path = eval_dir / "per_run_metrics.csv"
    if not per_run_path.exists():
        raise FileNotFoundError(
            f"{per_run_path} not found. Run eval/scripts/run_classification_eval.py first."
        )
    per_run = pd.read_csv(per_run_path)
    if per_run.empty:
        print(f"[WARN] {per_run_path} is empty; nothing to do.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    digest_chunks: list[str] = []

    for metric_col, metric_label in metrics:
        if metric_col not in per_run.columns:
            print(f"[WARN] metric {metric_col!r} not in per_run_metrics; skipping.")
            continue
        pivot = build_pivot(per_run, metric=metric_col)
        csv_path = out_dir / f"baseline_pivot_{metric_col}.csv"
        pivot.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}  ({len(pivot)} baselines × {pivot.shape[1] - 3} task cols)")
        digest_chunks.append(render_markdown(pivot, title=metric_label))

    digest = "\n".join(digest_chunks)
    digest_path = out_dir / "baseline_tables.md"
    digest_path.write_text(digest, encoding="utf-8")
    print(f"  wrote {digest_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    metric_specs = (
        [(m, m) for m in args.metrics] if args.metrics else list(DEFAULT_METRICS)
    )
    _run(eval_dir, out_dir, metric_specs)


if __name__ == "__main__":
    main()
