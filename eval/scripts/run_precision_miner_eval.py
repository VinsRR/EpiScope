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
    load_precision_miner_cases,
    run_precision_miner_cases,
    summarize_precision_miner_runs,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run precision-miner evaluation against EpiScope.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset of precision-miner cases.")
    parser.add_argument("--out-dir", required=True, help="Directory for evaluation outputs.")

    parser.add_argument("--loader", default="unstructured")
    parser.add_argument("--embed-model", default="gemini-embedding-001")
    parser.add_argument("--chunker", default="paragraph")
    parser.add_argument("--min-chunk-size", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)

    parser.add_argument("--index-dir", default=".episcope_index")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="episcope")
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

    cases = load_precision_miner_cases(args.dataset)
    pipeline_config = RagPipelineConfig(
        loader=args.loader,
        embed_model=args.embed_model,
        chunker=args.chunker,
        min_chunk_size=args.min_chunk_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_dir=args.index_dir,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        top_k=args.top_k,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        continue_on_error=not args.stop_on_error,
    )

    runs = run_precision_miner_cases(cases, pipeline_config)
    write_jsonl((run.to_dict() for run in runs), out_dir / "case_runs.jsonl")
    write_csv(pd.DataFrame([run.to_dict() for run in runs]), out_dir / "case_runs.csv")

    manifest = {
        "dataset": str(Path(args.dataset).resolve()),
        "out_dir": str(out_dir),
        "case_count": len(cases),
        "pipeline_config": asdict(pipeline_config),
        "summary": summarize_precision_miner_runs(runs),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
