from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from episcope.episcope import ChunkerKind, LoaderKind, _build_chunker, _load_path

from .io import dump_generated_queries
from .models import GeneratedQueryReviewRecord, RagPipelineConfig


def _require_ragas_testset():
    try:
        from ragas.testset import TestsetGenerator
    except ImportError:
        from ragas.testset.synthesizers.generate import TestsetGenerator  # type: ignore
    return TestsetGenerator


def _load_chunk_texts(path: str | Path, config: RagPipelineConfig) -> list[str]:
    papers = _load_path(Path(path), loader_kind=LoaderKind(config.loader))
    chunker = _build_chunker(
        ChunkerKind(config.chunker),
        min_chunk_size=config.min_chunk_size,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks: list[str] = []
    for paper in papers:
        if paper.metadata.abstract:
            abstract = paper.metadata.abstract.strip()
            if abstract:
                chunks.append(abstract)
        for section in paper.sections:
            for text in chunker.chunk(section.content):
                text = text.strip()
                if text:
                    chunks.append(text)
    return chunks


def _build_testset_models(
    *,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str,
    embedding_model: str,
    api_key: str | None = None,
    api_base: str | None = None,
) -> tuple[Any, Any]:
    from .ragas_adapter import _build_evaluator_embeddings, _build_evaluator_llm
    from .models import RagasEvaluatorConfig

    config = RagasEvaluatorConfig(
        metric_names=[],
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key,
        api_base=api_base,
        base_url=api_base,
    )
    llm = _build_evaluator_llm(config)
    embeddings = _build_evaluator_embeddings(config)
    if llm is None or embeddings is None:
        raise ValueError("Both evaluator LLM and embeddings must be configured for testset generation.")
    return llm, embeddings


def generate_testset_candidates(
    *,
    path: str | Path,
    pipeline_config: RagPipelineConfig,
    llm_provider: str,
    llm_model: str,
    embedding_provider: str,
    embedding_model: str,
    testset_size: int,
    out_jsonl: str | Path,
    out_csv: str | Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> list[GeneratedQueryReviewRecord]:
    TestsetGenerator = _require_ragas_testset()
    chunk_texts = _load_chunk_texts(path, pipeline_config)
    llm, embeddings = _build_testset_models(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key,
        api_base=api_base,
    )

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    testset = generator.generate_with_chunks(chunks=chunk_texts, testset_size=testset_size)

    if out_csv is not None:
        output_csv = Path(out_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        testset.to_csv(str(output_csv))

    records: list[GeneratedQueryReviewRecord] = []
    frame = testset.to_pandas()
    root_path = str(Path(path).resolve())
    for row in frame.to_dict(orient="records"):
        records.append(
            GeneratedQueryReviewRecord(
                schema_version="1",
                query_id=uuid.uuid4().hex,
                source_kind="ragas_generate_with_chunks",
                paper_path=root_path,
                user_input=str(row.get("user_input", "") or ""),
                reference=str(row.get("reference", "") or ""),
                reference_contexts=list(row.get("reference_contexts", []) or []),
                persona_name=row.get("persona_name"),
                synthesizer_name=row.get("synthesizer_name"),
                status="pending_review",
            )
        )

    dump_generated_queries(records, out_jsonl)
    return records
