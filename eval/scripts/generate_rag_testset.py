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

from eval.common.config import load_json_config
from eval.ragas import generate_testset_candidates
from eval.ragas.models import RagPipelineConfig


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate an explorer-style RAGAS testset from EpiScope chunks.")
    parser.add_argument(
        "--config",
        help="Optional JSON config file. Command-line flags override config values.",
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--doc-id", help="Indexed document id whose stored chunks should be used.")
    source_group.add_argument("--path", help="File or directory to chunk with EpiScope.")
    parser.add_argument("--out-jsonl", help="Where to store generated review records.")
    parser.add_argument(
        "--out-cases-jsonl",
        help="Optional JSONL export in SimpleRagQaCase format for direct use with run_rag_eval.py.",
    )
    parser.add_argument("--out-csv", help="Optional raw CSV export from RAGAS.")
    parser.add_argument("--testset-size", type=int, default=10)
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Optional cap on source chunks before RAGAS transforms. Useful for fast smoke tests.",
    )

    parser.add_argument("--loader", default="unstructured")
    parser.add_argument("--chunker", default="paragraph")
    parser.add_argument("--min-chunk-size", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--index-backend", default="file")
    parser.add_argument("--index-dir", default=".episcope_index")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="episcope")

    parser.add_argument("--generator-llm-provider")
    parser.add_argument("--generator-llm-model")
    parser.add_argument("--critic-llm-provider")
    parser.add_argument("--critic-llm-model")
    parser.add_argument("--generator-embedding-provider")
    parser.add_argument("--generator-embedding-model")
    parser.add_argument("--generator-api-key")
    parser.add_argument("--generator-api-base")
    parser.add_argument("--generator-max-tokens", type=int)
    parser.add_argument(
        "--generator-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Reasoning effort passed through to OpenAI-compatible generator calls.",
    )
    parser.add_argument("--max-entities-per-chunk", type=int, default=10)
    parser.add_argument("--max-themes-per-chunk", type=int, default=10)
    parser.add_argument("--simple-ratio", type=float, default=0.5)
    parser.add_argument("--reasoning-ratio", type=float, default=0.25)
    parser.add_argument("--multi-context-ratio", type=float, default=0.25)
    return parser


def parse_args_with_config() -> object:
    parser = build_parser()
    initial_args, _ = parser.parse_known_args()
    if getattr(initial_args, "config", None):
        config = load_json_config(
            initial_args.config,
            allowed_keys={action.dest for action in parser._actions if action.dest != "help"},
        )
        parser.set_defaults(**config)
    args = parser.parse_args()
    missing: list[str] = []
    if not args.doc_id and not args.path:
        missing.append("one of --doc-id or --path")
    if not args.out_jsonl:
        missing.append("--out-jsonl")
    if not args.generator_llm_provider:
        missing.append("--generator-llm-provider")
    if not args.generator_llm_model:
        missing.append("--generator-llm-model")
    if not args.generator_embedding_provider:
        missing.append("--generator-embedding-provider")
    if not args.generator_embedding_model:
        missing.append("--generator-embedding-model")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args_with_config()

    pipeline_config = RagPipelineConfig(
        loader=args.loader,
        chunker=args.chunker,
        min_chunk_size=args.min_chunk_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_backend=args.index_backend,
        index_dir=args.index_dir,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
    )

    generate_testset_candidates(
        path=args.path,
        doc_id=args.doc_id,
        pipeline_config=pipeline_config,
        llm_provider=args.generator_llm_provider,
        llm_model=args.generator_llm_model,
        critic_llm_provider=args.critic_llm_provider,
        critic_llm_model=args.critic_llm_model,
        embedding_provider=args.generator_embedding_provider,
        embedding_model=args.generator_embedding_model,
        testset_size=args.testset_size,
        out_jsonl=args.out_jsonl,
        out_cases_jsonl=args.out_cases_jsonl,
        out_csv=args.out_csv,
        api_key=args.generator_api_key,
        api_base=args.generator_api_base,
        generator_max_tokens=args.generator_max_tokens,
        generator_reasoning_effort=args.generator_reasoning_effort,
        max_chunks=args.max_chunks,
        max_entities_per_chunk=args.max_entities_per_chunk,
        max_themes_per_chunk=args.max_themes_per_chunk,
        simple_ratio=args.simple_ratio,
        reasoning_ratio=args.reasoning_ratio,
        multi_context_ratio=args.multi_context_ratio,
    )


if __name__ == "__main__":
    main()
