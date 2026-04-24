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
from eval.ragas import evaluate_case_runs, load_simple_rag_qa_cases, run_cases
from eval.ragas.io import dump_case_runs
from eval.ragas.models import RagPipelineConfig, RagasEvaluatorConfig
from eval.ragas.ragas_adapter import summarize_ragas_scores


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run QA-style RAG evaluation against EpiScope.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset of SimpleRagQaCase records.")
    parser.add_argument("--out-dir", required=True, help="Directory for evaluation outputs.")

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

    parser.add_argument("--llm-provider", default="nollm")
    parser.add_argument("--llm-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Metric name to run. Repeatable. Defaults to the standard RAG metric set.",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_simple_rag_qa_cases(args.dataset)
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

    runs = run_cases(cases, pipeline_config)
    dump_case_runs(runs, out_dir / "case_runs.jsonl")
    write_csv(pd.DataFrame([run.to_dict() for run in runs]), out_dir / "case_runs.csv")

    successful_runs = [run for run in runs if run.run_error is None]
    local_summary = {
        "row_count": len(runs),
        "successful_runs": len(successful_runs),
        "failed_runs": len(runs) - len(successful_runs),
        "exact_match_mean": float(
            pd.Series([run.exact_match for run in successful_runs if run.exact_match is not None]).mean()
        )
        if any(run.exact_match is not None for run in successful_runs)
        else None,
        "reference_context_f1_mean": float(
            pd.Series(
                [
                    run.reference_context_f1
                    for run in successful_runs
                    if run.reference_context_f1 is not None
                ]
            ).mean()
        )
        if any(run.reference_context_f1 is not None for run in successful_runs)
        else None,
    }

    manifest = {
        "dataset": str(Path(args.dataset).resolve()),
        "out_dir": str(out_dir),
        "case_count": len(cases),
        "pipeline_config": asdict(pipeline_config),
        "ragas_requested": not args.skip_ragas,
        "local_summary": local_summary,
    }

    if args.skip_ragas:
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return

    evaluator_config = RagasEvaluatorConfig(
        metric_names=args.metrics or RagasEvaluatorConfig().metric_names,
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

    try:
        scores = evaluate_case_runs(runs, evaluator_config)
        if not scores.empty:
            write_csv(scores, out_dir / "ragas_scores.csv")
        summary = summarize_ragas_scores(scores, evaluator_config.metric_names)
    except Exception as exc:
        manifest["ragas_error"] = str(exc)
        summary = {}
    manifest["ragas_config"] = asdict(evaluator_config)
    manifest["ragas_summary"] = summary
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
