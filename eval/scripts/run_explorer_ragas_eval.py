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

from eval.common.config import load_json_config
from eval.common.io import write_csv
from eval.ragas import evaluate_case_runs, generate_explorer_testset, run_cases
from eval.ragas.io import dump_case_runs
from eval.ragas.models import RagPipelineConfig, RagasEvaluatorConfig
from eval.ragas.ragas_adapter import summarize_ragas_scores


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate a synthetic explorer benchmark with RAGAS, then run and score the EpiScope explorer pipeline."
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. Command-line flags override config values.",
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--doc-id", help="Indexed document id whose stored chunks should seed generation.")
    source_group.add_argument("--path", help="File or directory used as the explorer knowledge base.")
    parser.add_argument("--out-dir", help="Directory where generated assets and eval outputs are written.")
    parser.add_argument("--testset-size", type=int, default=10)

    parser.add_argument("--loader", default="unstructured")
    parser.add_argument("--embed-model", default="gemini-embedding-001")
    parser.add_argument("--chunker", default="paragraph")
    parser.add_argument("--min-chunk-size", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)

    parser.add_argument("--index-backend", default="file")
    parser.add_argument("--index-dir", default=".episcope_index")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="episcope")
    parser.add_argument("--retrieval-mode", default="dense_only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.0)

    parser.add_argument("--generator-llm-provider")
    parser.add_argument("--generator-llm-model")
    parser.add_argument("--critic-llm-provider")
    parser.add_argument("--critic-llm-model")
    parser.add_argument("--generator-embedding-provider")
    parser.add_argument("--generator-embedding-model")
    parser.add_argument("--generator-api-key")
    parser.add_argument("--generator-api-base")
    parser.add_argument("--simple-ratio", type=float, default=0.5)
    parser.add_argument("--reasoning-ratio", type=float, default=0.25)
    parser.add_argument("--multi-context-ratio", type=float, default=0.25)

    parser.add_argument("--llm-provider", default="nollm")
    parser.add_argument("--llm-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Metric name to run. Defaults to faithfulness + response_relevancy.",
    )
    parser.add_argument("--evaluator-llm-provider")
    parser.add_argument("--evaluator-llm-model")
    parser.add_argument("--evaluator-embedding-provider")
    parser.add_argument("--evaluator-embedding-model")
    parser.add_argument("--evaluator-api-key")
    parser.add_argument("--evaluator-api-base")
    parser.add_argument("--evaluator-api-version")
    parser.add_argument("--raise-ragas-exceptions", action="store_true")
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
    if not args.out_dir:
        missing.append("--out-dir")
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
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = RagPipelineConfig(
        loader=args.loader,
        embed_model=args.embed_model,
        chunker=args.chunker,
        min_chunk_size=args.min_chunk_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_backend=args.index_backend,
        index_dir=args.index_dir,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        continue_on_error=not args.stop_on_error,
    )

    generated = generate_explorer_testset(
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
        out_review_jsonl=out_dir / "generated_queries.jsonl",
        out_cases_jsonl=out_dir / "generated_cases.jsonl",
        out_csv=out_dir / "generated_testset.csv",
        api_key=args.generator_api_key,
        api_base=args.generator_api_base,
        simple_ratio=args.simple_ratio,
        reasoning_ratio=args.reasoning_ratio,
        multi_context_ratio=args.multi_context_ratio,
    )

    runs = run_cases(generated.qa_cases, pipeline_config)
    dump_case_runs(runs, out_dir / "case_runs.jsonl")
    runs_df = pd.DataFrame([run.to_dict() for run in runs])
    write_csv(runs_df, out_dir / "case_runs.csv")

    evaluator_config = RagasEvaluatorConfig(
        metric_names=args.metrics or ["faithfulness", "response_relevancy"],
        llm_provider=args.evaluator_llm_provider,
        llm_model=args.evaluator_llm_model,
        embedding_provider=args.evaluator_embedding_provider,
        embedding_model=args.evaluator_embedding_model,
        api_key=args.evaluator_api_key,
        api_base=args.evaluator_api_base,
        base_url=args.evaluator_api_base,
        api_version=args.evaluator_api_version,
        raise_exceptions=args.raise_ragas_exceptions,
    )

    manifest = {
        "path": str(Path(args.path).resolve()) if args.path else None,
        "doc_id": args.doc_id,
        "out_dir": str(out_dir),
        "pipeline_config": asdict(pipeline_config),
        "testset_size": args.testset_size,
        "generated_case_count": len(generated.qa_cases),
        "ragas_config": asdict(evaluator_config),
    }

    try:
        scores = evaluate_case_runs(runs, evaluator_config)
        if not scores.empty:
            write_csv(scores, out_dir / "ragas_scores.csv")
            merged = runs_df.merge(scores, on=["case_id", "paper_path", "paper_id"], how="left")
            write_csv(merged, out_dir / "explorer_eval_rows.csv")
        manifest["ragas_summary"] = summarize_ragas_scores(scores, evaluator_config.metric_names)
    except Exception as exc:
        manifest["ragas_error"] = str(exc)

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
