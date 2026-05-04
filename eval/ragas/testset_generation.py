from __future__ import annotations

import uuid
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import pandas as pd

from episcope.episcope import (
    ChunkerKind,
    IndexBackend,
    LoaderKind,
    _build_chunker,
    _build_vector_db,
    _load_path,
)
from episcope.rag.indexing.chunking import chunk_paper

from .io import dump_generated_queries, write_jsonl
from .models import GeneratedQueryReviewRecord, RagPipelineConfig, SimpleRagQaCase


def _require_ragas_testset():
    try:
        from ragas.testset import TestsetGenerator
    except ImportError:
        from ragas.testset.synthesizers.generate import TestsetGenerator  # type: ignore
    return TestsetGenerator


def _optional_langchain_document():
    try:
        from langchain_core.documents import Document
    except ImportError:
        return None
    return Document


def _enum(enum_cls: Any, value: str):
    try:
        return enum_cls(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Unsupported value {value!r}. Expected one of: {supported}") from exc


def _chunk_to_document(chunk: dict[str, Any], Document: Any | None) -> Any:
    text = str(chunk.get("text", "") or "").strip()
    if not text:
        return None
    if Document is None:
        return text
    return Document(
        page_content=text,
        metadata={
            "paper_id": chunk.get("paper_id"),
            "section_title": chunk.get("section_title", ""),
            "section_type": chunk.get("section_type", ""),
            "is_metadata": bool(chunk.get("is_metadata", False)),
            "chunk_id": chunk.get("id"),
        },
    )


def _load_chunk_documents(path: str | Path, config: RagPipelineConfig) -> list[Any]:
    papers = _load_path(Path(path), loader_kind=LoaderKind(config.loader))
    chunker = _build_chunker(
        ChunkerKind(config.chunker),
        min_chunk_size=config.min_chunk_size,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    Document = _optional_langchain_document()
    chunks: list[Any] = []
    for paper in papers:
        for chunk in chunk_paper(chunker, paper.sections, paper.metadata, paper.paper_id):
            document = _chunk_to_document(chunk, Document)
            if document is not None:
                chunks.append(document)
    return chunks


def _load_stored_chunk_documents(doc_id: str, config: RagPipelineConfig) -> list[Any]:
    Document = _optional_langchain_document()
    vectordb = _build_vector_db(
        _enum(IndexBackend, config.index_backend),
        index_dir=Path(config.index_dir),
        qdrant_collection=config.qdrant_collection,
        qdrant_url=config.qdrant_url,
    )
    points = list(vectordb.get_points(namespace=doc_id))
    if not points:
        raise ValueError(
            f"No indexed chunks were found for doc_id={doc_id!r}. "
            "Check the index backend, index directory/collection, and document id."
        )

    chunks: list[Any] = []
    for point in points:
        document = _chunk_to_document(point, Document)
        if document is not None:
            chunks.append(document)
    if not chunks:
        raise ValueError(f"Indexed chunks for doc_id={doc_id!r} were found but contained no text.")
    return chunks


def _build_testset_models(
    *,
    llm_provider: str,
    llm_model: str,
    critic_llm_provider: str | None,
    critic_llm_model: str | None,
    embedding_provider: str,
    embedding_model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    generator_max_tokens: int | None = None,
    generator_reasoning_effort: str | None = None,
) -> tuple[Any, Any, Any | None]:
    from .ragas_adapter import _build_evaluator_embeddings, _build_evaluator_llm
    from .models import RagasEvaluatorConfig

    resolved_generator_max_tokens = (
        generator_max_tokens
        if generator_max_tokens is not None
        else _default_testset_max_tokens(llm_provider, llm_model)
    )
    resolved_generator_reasoning_effort = (
        generator_reasoning_effort
        if generator_reasoning_effort is not None
        else _default_testset_reasoning_effort(llm_provider, llm_model)
    )
    resolved_critic_max_tokens = (
        generator_max_tokens
        if generator_max_tokens is not None
        else _default_testset_max_tokens(critic_llm_provider, critic_llm_model)
    )
    resolved_critic_reasoning_effort = (
        generator_reasoning_effort
        if generator_reasoning_effort is not None
        else _default_testset_reasoning_effort(critic_llm_provider, critic_llm_model)
    )

    generator_config = RagasEvaluatorConfig(
        metric_names=[],
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key,
        api_base=api_base,
        base_url=api_base,
        max_tokens=resolved_generator_max_tokens,
        reasoning_effort=resolved_generator_reasoning_effort,
    )
    critic_config = RagasEvaluatorConfig(
        metric_names=[],
        llm_provider=critic_llm_provider,
        llm_model=critic_llm_model,
        api_key=api_key,
        api_base=api_base,
        base_url=api_base,
        max_tokens=resolved_critic_max_tokens,
        reasoning_effort=resolved_critic_reasoning_effort,
    )

    llm = _build_evaluator_llm(generator_config)
    critic_llm = (
        _build_evaluator_llm(critic_config)
        if critic_llm_provider and critic_llm_model
        else None
    )
    embeddings = _build_evaluator_embeddings(generator_config)
    if llm is None or embeddings is None:
        raise ValueError(
            "Both generator LLM and embeddings must be configured for testset generation."
        )
    return llm, embeddings, critic_llm


def _is_google_provider(provider: str | None) -> bool:
    return (provider or "").strip().lower() in {"gemini", "google"}


def _default_testset_max_tokens(provider: str | None, model: str | None) -> int | None:
    if _is_google_provider(provider) and model:
        return 8192
    return None


def _default_testset_reasoning_effort(
    provider: str | None,
    model: str | None,
) -> str | None:
    model_name = (model or "").strip().lower()
    if (
        _is_google_provider(provider)
        and model_name.startswith("gemini-2.5")
        and "pro" not in model_name
    ):
        return "none"
    return None


def _require_default_query_distribution():
    try:
        from ragas.testset.synthesizers import default_query_distribution
    except ImportError:
        return None
    return default_query_distribution


def _build_query_distribution(
    generator_llm: Any,
    *,
    simple_ratio: float | None = None,
    reasoning_ratio: float | None = None,
    multi_context_ratio: float | None = None,
):
    requested = [simple_ratio, reasoning_ratio, multi_context_ratio]
    if all(value is None for value in requested):
        return None

    simple_ratio = float(simple_ratio or 0.0)
    reasoning_ratio = float(reasoning_ratio or 0.0)
    multi_context_ratio = float(multi_context_ratio or 0.0)
    total = simple_ratio + reasoning_ratio + multi_context_ratio
    if total <= 0:
        raise ValueError("At least one query-distribution weight must be > 0.")

    targets = {
        "simple": simple_ratio / total,
        "reasoning": reasoning_ratio / total,
        "multi_context": multi_context_ratio / total,
    }

    default_distribution_builder = _require_default_query_distribution()
    if default_distribution_builder is not None:
        default_distribution = list(default_distribution_builder(generator_llm))
        grouped: dict[str, list[tuple[Any, float]]] = {
            "simple": [],
            "reasoning": [],
            "multi_context": [],
        }
        for synthesizer, weight in default_distribution:
            name = type(synthesizer).__name__.lower()
            if "singlehop" in name or "single_hop" in name:
                bucket = "simple"
            elif "abstract" in name or "reason" in name:
                bucket = "reasoning"
            else:
                bucket = "multi_context"
            grouped[bucket].append((synthesizer, float(weight)))

        adjusted: list[tuple[Any, float]] = []
        for bucket, items in grouped.items():
            target = targets[bucket]
            if target == 0.0 or not items:
                continue
            weight_sum = sum(weight for _, weight in items)
            if weight_sum <= 0:
                per_item = target / len(items)
                adjusted.extend((synthesizer, per_item) for synthesizer, _ in items)
            else:
                adjusted.extend(
                    (synthesizer, target * (weight / weight_sum))
                    for synthesizer, weight in items
                )
        return adjusted

    try:
        from ragas.testset.evolutions import multi_context, reasoning, simple  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not build a query distribution for this ragas version."
        ) from exc

    return {
        simple: targets["simple"],
        reasoning: targets["reasoning"],
        multi_context: targets["multi_context"],
    }


def _query_types_from_ratios(
    *,
    simple_ratio: float | None,
    reasoning_ratio: float | None,
    multi_context_ratio: float | None,
) -> set[str]:
    ratios = {
        "simple": simple_ratio,
        "reasoning": reasoning_ratio,
        "multi_context": multi_context_ratio,
    }
    if all(value is None for value in ratios.values()):
        return set(ratios)
    return {name for name, value in ratios.items() if float(value or 0.0) > 0.0}


def build_transforms(
    llm,
    embeddings,
    *,
    query_types: set[str] | None = None,
    max_num_entities: int = 10,
    max_num_themes: int = 10,
):
    # Custom pre-chunked transforms: preserve the summary + summary_embedding fields
    # that later RAGAS persona generation relies on, while skipping CustomNodeFilter
    # due to the upstream pre-chunked-summary bug tracked in ragas#2680.
    from ragas.testset.graph import NodeType
    from ragas.testset.transforms.engine import Parallel
    from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
    from ragas.testset.transforms.extractors.llm_based import ThemesExtractor, NERExtractor
    from ragas.testset.transforms.relationship_builders import (
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
    )

    def filter_chunks(node):
        return node.type == NodeType.CHUNK

    query_types = query_types or {"simple", "reasoning", "multi_context"}
    include_entities = "simple" in query_types or "multi_context" in query_types
    include_themes = "reasoning" in query_types
    include_similarity = "reasoning" in query_types
    include_overlap = "multi_context" in query_types

    extractors = [
        EmbeddingExtractor(
            embedding_model=embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=filter_chunks,
        )
    ]
    if include_themes:
        extractors.append(
            ThemesExtractor(
                llm=llm,
                filter_nodes=filter_chunks,
                max_num_themes=max_num_themes,
            )
        )
    if include_entities:
        extractors.append(
            NERExtractor(
                llm=llm,
                filter_nodes=filter_chunks,
                max_num_entities=max_num_entities,
            )
        )

    transforms: list[Any] = [
        SummaryExtractor(llm=llm, filter_nodes=filter_chunks),
        Parallel(*extractors),
    ]

    relationship_builders = []
    if include_similarity:
        relationship_builders.append(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.7,
                filter_nodes=filter_chunks,
            )
        )
    if include_overlap:
        relationship_builders.append(
            OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)
        )
    if relationship_builders:
        if len(relationship_builders) == 1:
            transforms.append(relationship_builders[0])
        else:
            transforms.append(Parallel(*relationship_builders))

    return transforms


def _limit_chunk_docs(chunk_docs: list[Any], max_chunks: int | None) -> list[Any]:
    if max_chunks is None:
        return chunk_docs
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0 when provided.")
    return chunk_docs[:max_chunks]

@dataclass
class GeneratedExplorerTestset:
    frame: pd.DataFrame
    review_records: list[GeneratedQueryReviewRecord]
    qa_cases: list[SimpleRagQaCase]


def _build_review_records(frame: pd.DataFrame, *, root_path: str) -> list[GeneratedQueryReviewRecord]:
    records: list[GeneratedQueryReviewRecord] = []
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
    return records


def _build_review_records_for_doc_id(frame: pd.DataFrame, *, doc_id: str) -> list[GeneratedQueryReviewRecord]:
    records: list[GeneratedQueryReviewRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            GeneratedQueryReviewRecord(
                schema_version="1",
                query_id=uuid.uuid4().hex,
                source_kind="ragas_generate_with_chunks",
                paper_id=doc_id,
                user_input=str(row.get("user_input", "") or ""),
                reference=str(row.get("reference", "") or ""),
                reference_contexts=list(row.get("reference_contexts", []) or []),
                persona_name=row.get("persona_name"),
                synthesizer_name=row.get("synthesizer_name"),
                status="pending_review",
            )
        )
    return records


def _build_simple_rag_qa_cases(
    records: list[GeneratedQueryReviewRecord],
) -> list[SimpleRagQaCase]:
    cases: list[SimpleRagQaCase] = []
    for record in records:
        cases.append(
            SimpleRagQaCase(
                schema_version="1",
                case_id=f"generated_{record.query_id}",
                paper_path=record.paper_path,
                paper_id=record.paper_id,
                user_input=record.user_input,
                reference=record.reference,
                reference_contexts=list(record.reference_contexts),
                tags=["generated", "explorer"],
                metadata={
                    "source_kind": record.source_kind,
                    "persona_name": record.persona_name,
                    "synthesizer_name": record.synthesizer_name,
                },
                notes="Synthetic QA case generated by RAGAS from EpiScope chunks.",
            )
        )
    return cases


def generate_explorer_testset(
    *,
    path: str | Path | None = None,
    doc_id: str | None = None,
    pipeline_config: RagPipelineConfig,
    llm_provider: str,
    llm_model: str,
    critic_llm_provider: str | None,
    critic_llm_model: str | None,
    embedding_provider: str,
    embedding_model: str,
    testset_size: int,
    out_review_jsonl: str | Path | None = None,
    out_cases_jsonl: str | Path | None = None,
    out_csv: str | Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    generator_max_tokens: int | None = None,
    generator_reasoning_effort: str | None = None,
    max_chunks: int | None = None,
    max_entities_per_chunk: int = 10,
    max_themes_per_chunk: int = 10,
    simple_ratio: float | None = None,
    reasoning_ratio: float | None = None,
    multi_context_ratio: float | None = None,
) -> GeneratedExplorerTestset:
    if bool(path is not None) == bool(doc_id is not None):
        raise ValueError("Provide exactly one of `path` or `doc_id`.")
    TestsetGenerator = _require_ragas_testset()
    if doc_id is not None:
        chunk_docs = _load_stored_chunk_documents(doc_id, pipeline_config)
    else:
        chunk_docs = _load_chunk_documents(path, pipeline_config)
    chunk_docs = _limit_chunk_docs(chunk_docs, max_chunks)
    llm, embeddings, critic_llm = _build_testset_models(
        llm_provider=llm_provider,
        llm_model=llm_model,
        critic_llm_provider=critic_llm_provider,
        critic_llm_model=critic_llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        api_key=api_key,
        api_base=api_base,
        generator_max_tokens=generator_max_tokens,
        generator_reasoning_effort=generator_reasoning_effort,
    )

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    query_distribution = _build_query_distribution(
        llm,
        simple_ratio=simple_ratio,
        reasoning_ratio=reasoning_ratio,
        multi_context_ratio=multi_context_ratio,
    )

    generate_method = getattr(generator, "generate_with_chunks", None)
    if generate_method is None:
        generate_method = getattr(generator, "generate_with_langchain_docs", None)
    if generate_method is None:
        raise AttributeError(
            "This ragas TestsetGenerator does not expose a supported generation method."
        )

    signature = inspect.signature(generate_method)
    generate_kwargs: dict[str, Any] = {"testset_size": testset_size}

    if "chunks" in signature.parameters:
        generate_kwargs["chunks"] = chunk_docs
    elif "docs" in signature.parameters:
        generate_kwargs["docs"] = chunk_docs
    elif "documents" in signature.parameters:
        generate_kwargs["documents"] = chunk_docs
    else:
        raise TypeError(
            "Unsupported testset-generation signature: expected chunks, docs, or documents parameter."
        )

    if "transforms" in signature.parameters and "chunks" in generate_kwargs:
        generate_kwargs["transforms"] = build_transforms(
            llm=critic_llm or llm,
            embeddings=embeddings,
            query_types=_query_types_from_ratios(
                simple_ratio=simple_ratio,
                reasoning_ratio=reasoning_ratio,
                multi_context_ratio=multi_context_ratio,
            ),
            max_num_entities=max_entities_per_chunk,
            max_num_themes=max_themes_per_chunk,
        )
    elif critic_llm is not None and "transforms_llm" in signature.parameters:
        generate_kwargs["transforms_llm"] = critic_llm

    if query_distribution is not None and "query_distribution" in signature.parameters:
        generate_kwargs["query_distribution"] = query_distribution

    testset = generate_method(**generate_kwargs)

    if out_csv is not None:
        output_csv = Path(out_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        testset.to_csv(str(output_csv))

    frame = testset.to_pandas()
    if doc_id is not None:
        review_records = _build_review_records_for_doc_id(frame, doc_id=doc_id)
    else:
        root_path = str(Path(path).resolve())
        review_records = _build_review_records(frame, root_path=root_path)
    qa_cases = _build_simple_rag_qa_cases(review_records)

    if out_review_jsonl is not None:
        dump_generated_queries(review_records, out_review_jsonl)
    if out_cases_jsonl is not None:
        write_jsonl((case.to_dict() for case in qa_cases), out_cases_jsonl)

    return GeneratedExplorerTestset(
        frame=frame,
        review_records=review_records,
        qa_cases=qa_cases,
    )


def generate_testset_candidates(
    *,
    path: str | Path | None = None,
    doc_id: str | None = None,
    pipeline_config: RagPipelineConfig,
    llm_provider: str,
    llm_model: str,
    critic_llm_provider: str | None = None,
    critic_llm_model: str | None = None,
    embedding_provider: str,
    embedding_model: str,
    testset_size: int,
    out_jsonl: str | Path,
    out_cases_jsonl: str | Path | None = None,
    out_csv: str | Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    generator_max_tokens: int | None = None,
    generator_reasoning_effort: str | None = None,
    max_chunks: int | None = None,
    max_entities_per_chunk: int = 10,
    max_themes_per_chunk: int = 10,
    simple_ratio: float | None = None,
    reasoning_ratio: float | None = None,
    multi_context_ratio: float | None = None,
) -> list[GeneratedQueryReviewRecord]:
    generated = generate_explorer_testset(
        path=path,
        doc_id=doc_id,
        pipeline_config=pipeline_config,
        llm_provider=llm_provider,
        llm_model=llm_model,
        critic_llm_provider=critic_llm_provider,
        critic_llm_model=critic_llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        testset_size=testset_size,
        out_review_jsonl=out_jsonl,
        out_cases_jsonl=out_cases_jsonl,
        out_csv=out_csv,
        api_key=api_key,
        api_base=api_base,
        generator_max_tokens=generator_max_tokens,
        generator_reasoning_effort=generator_reasoning_effort,
        max_chunks=max_chunks,
        max_entities_per_chunk=max_entities_per_chunk,
        max_themes_per_chunk=max_themes_per_chunk,
        simple_ratio=simple_ratio,
        reasoning_ratio=reasoning_ratio,
        multi_context_ratio=multi_context_ratio,
    )
    return generated.review_records
