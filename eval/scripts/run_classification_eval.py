from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.classification.error_analysis import (
    containment_pro_vs_flash,
    load_merged_predictions,
    per_run_metrics,
    pro_always_wrong,
    within_config_consistency,
)
from eval.classification.per_label import (
    per_label_metrics_for_files,
    summarize_per_label_metrics,
)
from eval.classification.summary import summarize_per_run_metrics
from eval.common.io import write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classification evaluation over existing result TSVs.")
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
        "--out-dir",
        default="eval_outputs/classification_baselines",
        help="Directory where evaluation CSVs will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="final_*.tsv",
        help="Glob pattern to use when discovering result files.",
    )
    parser.add_argument(
        "--containment-temperature",
        default="0.0",
        help="Temperature used for pro-vs-flash containment analysis.",
    )
    parser.add_argument(
        "--pro-model",
        default="gemini-2-5-pro",
        help="Model name used as the 'pro' reference in containment analyses.",
    )
    parser.add_argument(
        "--flash-model",
        default="gemini-2-5-flash",
        help="Model name used as the 'flash' reference in containment analyses.",
    )
    parser.add_argument(
        "--per-label",
        action="store_true",
        default=False,
        help=(
            "When set, also compute per-label metrics and write "
            "per_label_metrics.csv and per_label_summary.csv to --out-dir."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = per_run_metrics(
        ground_truth_path=args.ground_truth,
        run_roots=args.run_roots,
        pattern=args.pattern,
    )
    write_csv(per_run, out_dir / "per_run_metrics.csv")

    summary = summarize_per_run_metrics(per_run)
    write_csv(summary, out_dir / "summary_metrics.csv")

    merged = load_merged_predictions(
        ground_truth_path=args.ground_truth,
        run_roots=args.run_roots,
        pattern=args.pattern,
    )
    write_csv(within_config_consistency(merged), out_dir / "within_config_consistency.csv")
    write_csv(
        containment_pro_vs_flash(
            merged,
            temperature=args.containment_temperature,
            pro_model=args.pro_model,
            flash_model=args.flash_model,
        ),
        out_dir / "containment_pro_vs_flash.csv",
    )
    write_csv(
        pro_always_wrong(
            merged,
            temperature=args.containment_temperature,
            pro_model=args.pro_model,
        ),
        out_dir / "pro_always_wrong.csv",
    )

    if args.per_label:
        per_label = per_label_metrics_for_files(
            ground_truth_path=args.ground_truth,
            run_roots=args.run_roots,
            pattern=args.pattern,
        )
        write_csv(per_label, out_dir / "per_label_metrics.csv")
        write_csv(summarize_per_label_metrics(per_label), out_dir / "per_label_summary.csv")

    print(f"Wrote classification evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
