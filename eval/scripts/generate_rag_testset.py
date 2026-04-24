from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.ragas import generate_testset_candidates
from eval.ragas.models import RagPipelineConfig


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate RAGAS testset candidates from EpiScope chunks.")
    parser.add_argument("--path", required=True, help="File or directory to chunk with EpiScope.")
    parser.add_argument("--out-jsonl", required=True, help="Where to store generated review records.")
    parser.add_argument("--out-csv", help="Optional raw CSV export from RAGAS.")
    parser.add_argument("--testset-size", type=int, default=10)

    parser.add_argument("--loader", default="unstructured")
    parser.add_argument("--chunker", default="paragraph")
    parser.add_argument("--min-chunk-size", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)

    parser.add_argument("--generator-llm-provider", required=True)
    parser.add_argument("--generator-llm-model", required=True)
    parser.add_argument("--generator-embedding-provider", required=True)
    parser.add_argument("--generator-embedding-model", required=True)
    parser.add_argument("--generator-api-key")
    parser.add_argument("--generator-api-base")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline_config = RagPipelineConfig(
        loader=args.loader,
        chunker=args.chunker,
        min_chunk_size=args.min_chunk_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    generate_testset_candidates(
        path=args.path,
        pipeline_config=pipeline_config,
        llm_provider=args.generator_llm_provider,
        llm_model=args.generator_llm_model,
        embedding_provider=args.generator_embedding_provider,
        embedding_model=args.generator_embedding_model,
        testset_size=args.testset_size,
        out_jsonl=args.out_jsonl,
        out_csv=args.out_csv,
        api_key=args.generator_api_key,
        api_base=args.generator_api_base,
    )


if __name__ == "__main__":
    main()
