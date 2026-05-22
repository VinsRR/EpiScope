"""Compute paired significance statistics for classification method pairs.

Reads the same result-TSV layout as ``run_classification_eval.py``, builds
per-paper score vectors, and reports paired bootstrap CIs on Jaccard
differences plus McNemar's test on exact-match indicators, with
Holm-Bonferroni correction across the pairs on each task. See
``eval/classification/significance.py`` for the underlying primitives and
``eval/classification/PAPER_NARRATIVE.md`` for the role this output plays in
§6.1.2 and §6.1.3 of the paper.

Examples
--------

  # EpiScope (flash) vs each ablation rung, three pairs per task:
  python eval/scripts/run_classification_significance.py \\
      --ground-truth sampled_papers_full.csv \\
      --run-root outputs/baselines \\
      --pair gemini-2-5-flash:metadata-llm-gemini-gemini-2.5-flash \\
      --pair gemini-2-5-flash:random-chunk-llm-gemini-gemini-2.5-flash-k_10-seed_13 \\
      --pair gemini-2-5-flash:prototype_similarity-emb_allenai-specter \\
      --out eval_outputs/classification/significance_ablation.csv

  # Method-cols = (model, temperature):
  python eval/scripts/run_classification_significance.py \\
      --ground-truth sampled_papers_full.csv \\
      --run-root outputs/episcope_runs \\
      --method-cols model,temperature \\
      --pair gemini-2-5-flash@0.0:gemini-2-5-flash@1.0 \\
      --out eval_outputs/classification/significance_temperature.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.classification.error_analysis import load_merged_predictions
from eval.classification.significance import pairwise_significance
from eval.common.io import write_csv


def _parse_method_key(raw: str, *, n_cols: int) -> str | tuple:
    """Parse a single method-key value.

    Multi-column keys are encoded with ``@`` (e.g. ``gemini-2-5-flash@0.0``).
    Single-column keys are returned as bare strings.
    """
    if n_cols == 1:
        if "@" in raw:
            raise ValueError(
                f"Method-cols has length 1 but key {raw!r} contains '@'. "
                f"Drop the @ or pass --method-cols accordingly."
            )
        return raw
    parts = raw.split("@")
    if len(parts) != n_cols:
        raise ValueError(
            f"Method key {raw!r} has {len(parts)} parts but method-cols has {n_cols}."
        )
    return tuple(parts)


def _parse_pair_arg(raw: str, *, n_cols: int) -> tuple:
    """Parse a ``A:B`` pair argument."""
    if raw.count(":") < 1:
        raise ValueError(f"--pair {raw!r} must contain ':' separating the two method keys.")
    # Split on the last ':' so multi-column keys like "model@temp" still parse.
    a, _, b = raw.rpartition(":")
    if not a or not b:
        raise ValueError(f"--pair {raw!r} must have a key on each side of ':'.")
    return _parse_method_key(a, n_cols=n_cols), _parse_method_key(b, n_cols=n_cols)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap + McNemar significance testing for classification baselines.",
    )
    parser.add_argument(
        "--ground-truth",
        default="sampled_papers_full.csv",
        help="Path to the benchmark TSV with gold labels.",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        required=True,
        help="Root directory to scan for final_*.tsv files. Repeatable.",
    )
    parser.add_argument(
        "--pattern",
        default="final_*.tsv",
        help="Glob pattern to use when discovering result files.",
    )
    parser.add_argument(
        "--method-cols",
        default="model",
        help=(
            "Comma-separated columns that together identify a method "
            "(default: model). For (model, temperature) pass "
            "'model,temperature' and encode pair keys with '@'."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        dest="pairs",
        required=True,
        help=(
            "Method pair to compare, encoded as 'A:B' where A and B are "
            "method keys. Repeatable; all pairs are corrected together per "
            "task. With multi-column method-cols, encode each key as "
            "'col1@col2' (e.g. 'gemini-2-5-flash@0.0')."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Restrict to one or more tasks (repeatable). Default: all tasks present.",
    )
    parser.add_argument(
        "--correctness-aggregator",
        choices=["majority", "all", "any"],
        default="majority",
        help="How to collapse per-run exact-match indicators per paper (default: majority).",
    )
    parser.add_argument(
        "--bootstrap-method",
        choices=["bca", "percentile"],
        default="bca",
        help="Bootstrap CI method (default: bca).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the CI (default: 0.05 → 95%% CI).",
    )
    parser.add_argument(
        "--correction",
        choices=["holm", "none"],
        default="holm",
        help="Multiple-comparison correction across pairs on each task (default: holm).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=13,
        help="Seed for the bootstrap resampler (default: 13).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the significance CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    method_cols = tuple(part.strip() for part in args.method_cols.split(",") if part.strip())
    if not method_cols:
        raise SystemExit("--method-cols must list at least one column.")
    pairs = [_parse_pair_arg(raw, n_cols=len(method_cols)) for raw in args.pairs]

    merged = load_merged_predictions(
        ground_truth_path=args.ground_truth,
        run_roots=args.run_roots,
        pattern=args.pattern,
    )
    if merged.empty:
        raise SystemExit(
            "No predictions discovered. Check --ground-truth and --run-root paths."
        )

    missing_cols = [col for col in method_cols if col not in merged.columns]
    if missing_cols:
        raise SystemExit(
            f"Method columns {missing_cols} are not present in the merged predictions. "
            f"Available columns: {sorted(merged.columns)}"
        )

    result = pairwise_significance(
        merged,
        pairs=pairs,
        method_cols=method_cols,
        tasks=args.tasks,
        correctness_aggregator=args.correctness_aggregator,
        B=args.bootstrap_iterations,
        alpha=args.alpha,
        bootstrap_method=args.bootstrap_method,
        correction=args.correction,
        random_state=args.random_state,
    )
    out_path = write_csv(result, args.out)
    if result.empty:
        print(
            f"Wrote {out_path} but it contains zero rows; check that the requested "
            f"pairs match values present in column(s) {method_cols}."
        )
    else:
        print(f"Wrote {out_path} ({len(result)} rows; {result['task'].nunique()} tasks)")


if __name__ == "__main__":
    main()
