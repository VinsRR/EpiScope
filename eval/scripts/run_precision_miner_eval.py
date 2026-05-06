from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.common.io import write_csv
from eval.ragas.io import write_jsonl
from eval.ragas.models import RagPipelineConfig
from eval.ragas.precision_miner_eval import (
    build_precision_miner_cases_from_references,
    build_precision_miner_references,
    precision_miner_agreement_metrics,
    run_precision_miner_cases,
    summarize_precision_miner_runs,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Run precision-miner evaluation against EpiScope. By default this "
            "builds automatic references from sampled_papers_full.csv and "
            "data-accessibility result TSVs, then runs only those sampled paper IDs."
        )
    )
    parser.add_argument(
        "--sampled-papers",
        default="sampled_papers_full.csv",
        help="TSV with sampled paper IDs and metadata.",
    )
    parser.add_argument(
        "--data-accessibility-root",
        action="append",
        dest="data_accessibility_roots",
        default=None,
        help="Root directory to scan for data-accessibility final TSVs. Repeatable.",
    )
    parser.add_argument(
        "--pattern",
        default="final_*.tsv",
        help="Glob pattern for data-accessibility outputs.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for evaluation outputs.")

    parser.add_argument("--index-backend", default="file")
    parser.add_argument("--index-dir", default=".episcope_index")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="episcope")
    parser.add_argument("--retrieval-mode", default="dense_only")
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--llm-provider", default="nollm")
    parser.add_argument("--llm-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_roots = args.data_accessibility_roots or ["outputs"]
    references = build_precision_miner_references(
        sampled_papers_path=args.sampled_papers,
        data_accessibility_roots=data_roots,
        pattern=args.pattern,
    )
    write_csv(references, out_dir / "references.csv")
    cases = build_precision_miner_cases_from_references(references)
    write_jsonl((case.to_dict() for case in cases), out_dir / "reference_cases.jsonl")

    pipeline_config = RagPipelineConfig(
        index_backend=args.index_backend,
        index_dir=args.index_dir,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        continue_on_error=not args.stop_on_error,
    )

    runs = run_precision_miner_cases(cases, pipeline_config)
    write_jsonl((run.to_dict() for run in runs), out_dir / "case_runs.jsonl")
    write_csv(pd.DataFrame([run.to_dict() for run in runs]), out_dir / "case_runs.csv")
    write_csv(precision_miner_agreement_metrics(runs), out_dir / "agreement_metrics.csv")

    manifest = {
        "sampled_papers": str(Path(args.sampled_papers).resolve()),
        "data_accessibility_roots": [str(Path(root).resolve()) for root in data_roots],
        "out_dir": str(out_dir),
        "reference_count": int(len(references)),
        "case_count": len(cases),
        "pipeline_config": asdict(pipeline_config),
        "summary": summarize_precision_miner_runs(runs),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
