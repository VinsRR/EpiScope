from __future__ import annotations

from pathlib import Path
from typing import Any

from episcope.episcope import (
    ChunkerKind,
    IndexBackend,
    LLMProvider,
    LoaderKind,
    RetrievalMode,
    _build_generator,
    _build_retriever,
    _build_vector_db,
    _json_ready,
    _transient_retriever_for_path,
)

from .local_metrics import exact_match_score, reference_context_prf
from .models import RagCaseRunResult, RagPipelineConfig, SimpleRagQaCase


def _enum(enum_cls: Any, value: str):
    try:
        return enum_cls(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Unsupported value {value!r}. Expected one of: {supported}") from exc


def run_case(case: SimpleRagQaCase, config: RagPipelineConfig) -> RagCaseRunResult:
    if case.paper_path is None and config.index_backend in {"file", "faiss"}:
        index_dir = Path(config.index_dir)
        if not index_dir.exists():
            raise ValueError(
                "Cases without paper_path require an existing indexed corpus. "
                f"Index directory {index_dir} does not exist."
            )

    retrieval_mode = _enum(RetrievalMode, config.retrieval_mode)
    llm_provider = _enum(LLMProvider, config.llm_provider)

    tempdir = None
    try:
        if case.paper_path is not None:
            if retrieval_mode != RetrievalMode.dense_only:
                raise ValueError("paper_path cases currently support only retrieval_mode='dense_only'.")

            tempdir, papers, retriever = _transient_retriever_for_path(
                Path(case.paper_path),
                loader_kind=_enum(LoaderKind, config.loader),
                embed_model=config.embed_model,
                chunker_kind=_enum(ChunkerKind, config.chunker),
                min_chunk_size=config.min_chunk_size,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            transient_paper_ids = {paper.paper_id for paper in papers}
            if case.paper_id is not None and case.paper_id not in transient_paper_ids:
                raise ValueError(
                    f"paper_id {case.paper_id!r} was not found under transient input {case.paper_path!r}."
                )
            paper_filter = case.paper_id or (papers[0].paper_id if len(papers) == 1 else None)
        else:
            vectordb = _build_vector_db(
                _enum(IndexBackend, config.index_backend),
                index_dir=Path(config.index_dir),
                qdrant_collection=config.qdrant_collection,
                qdrant_url=config.qdrant_url,
            )
            retriever = _build_retriever(vectordb, retrieval_mode=retrieval_mode)
            paper_filter = case.paper_id

        filter_payload = {"paper_id": paper_filter} if paper_filter else None
        results = list(
            retriever.retrieve(
                case.user_input,
                top_k=config.top_k,
                similarity_threshold=config.similarity_threshold,
                filter=filter_payload,
            )
        )
        retrieved_contexts = [chunk.text for chunk in results]

        generator = _build_generator(llm_provider, config.llm_model, config.temperature)
        provenance = generator.generate(results, question=case.user_input)
        response = provenance.answer

        precision, recall, f1 = reference_context_prf(
            retrieved_contexts,
            case.reference_contexts,
            threshold=config.local_context_match_threshold,
        )

        return RagCaseRunResult(
            case_id=case.case_id,
            paper_path=case.paper_path,
            paper_id=paper_filter or case.paper_id,
            user_input=case.user_input,
            reference=case.reference,
            response=response,
            reference_contexts=list(case.reference_contexts),
            retrieved_contexts=retrieved_contexts,
            retrieved_chunks=[_json_ready(chunk) for chunk in results],
            retrieval_count=len(results),
            tags=list(case.tags),
            metadata=dict(case.metadata),
            notes=case.notes,
            exact_match=exact_match_score(response, case.reference) if response is not None else None,
            reference_context_precision=precision,
            reference_context_recall=recall,
            reference_context_f1=f1,
        )
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def run_cases(
    cases: list[SimpleRagQaCase],
    config: RagPipelineConfig,
) -> list[RagCaseRunResult]:
    results: list[RagCaseRunResult] = []
    for case in cases:
        try:
            results.append(run_case(case, config))
        except Exception as exc:
            if not config.continue_on_error:
                raise
            results.append(
                RagCaseRunResult(
                    case_id=case.case_id,
                    paper_path=case.paper_path,
                    paper_id=case.paper_id,
                    user_input=case.user_input,
                    reference=case.reference,
                    response=None,
                    reference_contexts=list(case.reference_contexts),
                    tags=list(case.tags),
                    metadata=dict(case.metadata),
                    notes=case.notes,
                    run_error=str(exc),
                )
            )
    return results
