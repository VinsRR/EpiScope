from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from episcope.episcope import (
    ChunkerKind,
    LLMProvider,
    LoaderKind,
    PrecisionMinerKind,
    _build_generator,
    _build_precision_miner_config,
    _load_paper,
    _transient_retriever_for_path,
)
from episcope.workflows.precision_miner import PrecisionMiner

from .local_metrics import normalize_text, reference_context_prf
from .models import RagPipelineConfig
from .ragas_adapter import summarize_ragas_scores


def _enum(enum_cls: Any, value: str):
    try:
        return enum_cls(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Unsupported value {value!r}. Expected one of: {supported}") from exc


@dataclass
class PrecisionMinerReferenceContext:
    text: str
    section_type: Optional[str] = None
    supports_items: list[str] = field(default_factory=list)
    must_retrieve: bool = False


@dataclass
class PrecisionMinerReferenceItem:
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    url: Optional[str] = None
    url_aliases: list[str] = field(default_factory=list)
    expected_explanation_contains: list[str] = field(default_factory=list)
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_names(self) -> list[str]:
        return [self.canonical_name, *self.aliases]

    def all_urls(self) -> list[str]:
        return [item for item in [self.url, *self.url_aliases] if item]


@dataclass
class PrecisionMinerEvalCase:
    schema_version: str
    case_id: str
    miner_kind: str
    reference_answer: str
    reference_items: list[PrecisionMinerReferenceItem]
    paper_path: Optional[str] = None
    paper_id: Optional[str] = None
    reference_contexts: list[PrecisionMinerReferenceContext] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    rubrics: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrecisionMinerCaseRunResult:
    case_id: str
    miner_kind: str
    reference_answer: str
    response_description: Optional[str]
    paper_path: Optional[str] = None
    paper_id: Optional[str] = None
    reference_items: list[dict[str, Any]] = field(default_factory=list)
    predicted_items: list[dict[str, Any]] = field(default_factory=list)
    reference_contexts: list[dict[str, Any]] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    run_error: Optional[str] = None
    item_precision: Optional[float] = None
    item_recall: Optional[float] = None
    item_f1: Optional[float] = None
    required_item_recall: Optional[float] = None
    support_context_precision: Optional[float] = None
    support_context_recall: Optional[float] = None
    support_context_f1: Optional[float] = None
    must_retrieve_context_recall: Optional[float] = None
    explanation_coverage: Optional[float] = None
    description_exact_match: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_case_record(record: dict[str, Any], dataset_path: Path) -> PrecisionMinerEvalCase:
    missing = {
        key
        for key in ("schema_version", "case_id", "miner_kind", "reference_answer", "reference_items")
        if key not in record
    }
    if missing:
        raise ValueError(
            f"Missing required field(s) in {dataset_path}: {sorted(missing)} for case {record!r}"
        )

    paper_path = record.get("paper_path")
    if paper_path:
        candidate = Path(paper_path)
        if not candidate.is_absolute():
            candidate = (dataset_path.parent / candidate).resolve()
        paper_path = str(candidate)

    reference_items: list[PrecisionMinerReferenceItem] = []
    for item in record.get("reference_items", []) or []:
        canonical_name = item.get("canonical_name") or item.get("identifier") or item.get("canonical_citation")
        if not canonical_name:
            raise ValueError(f"Case {record.get('case_id')!r} contains a reference item without a canonical name.")
        metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "canonical_name",
                "identifier",
                "canonical_citation",
                "aliases",
                "url",
                "url_aliases",
                "expected_explanation_contains",
                "required",
            }
        }
        reference_items.append(
            PrecisionMinerReferenceItem(
                canonical_name=str(canonical_name),
                aliases=list(item.get("aliases", []) or []),
                url=item.get("url"),
                url_aliases=list(item.get("url_aliases", []) or []),
                expected_explanation_contains=list(item.get("expected_explanation_contains", []) or []),
                required=bool(item.get("required", True)),
                metadata=metadata,
            )
        )

    reference_contexts = [
        PrecisionMinerReferenceContext(
            text=str(item["text"]),
            section_type=item.get("section_type"),
            supports_items=list(item.get("supports_items", []) or []),
            must_retrieve=bool(item.get("must_retrieve", False)),
        )
        for item in (record.get("reference_contexts", []) or [])
    ]

    return PrecisionMinerEvalCase(
        schema_version=str(record["schema_version"]),
        case_id=str(record["case_id"]),
        miner_kind=str(record["miner_kind"]),
        paper_path=paper_path,
        paper_id=record.get("paper_id"),
        tags=list(record.get("tags", []) or []),
        reference_answer=str(record["reference_answer"]),
        reference_items=reference_items,
        reference_contexts=reference_contexts,
        rubrics=dict(record.get("rubrics", {}) or {}),
        notes=str(record.get("notes", "") or ""),
    )


def load_precision_miner_cases(path: str | Path) -> list[PrecisionMinerEvalCase]:
    from .io import read_jsonl

    dataset_path = Path(path).resolve()
    return [_load_case_record(record, dataset_path) for record in read_jsonl(dataset_path)]


def _best_name_match(predicted: dict[str, Any], reference: PrecisionMinerReferenceItem) -> bool:
    candidate_names = [
        normalize_text(predicted.get("name", "")),
        normalize_text(predicted.get("raw_text", "")),
    ]
    target_names = [normalize_text(item) for item in reference.all_names() if normalize_text(item)]
    if not target_names:
        return False
    return any(candidate and candidate in target for candidate in candidate_names for target in target_names) or any(
        target and target in candidate for candidate in candidate_names for target in target_names
    )


def _best_url_match(predicted: dict[str, Any], reference: PrecisionMinerReferenceItem) -> bool:
    predicted_url = normalize_text(predicted.get("url", ""))
    if not predicted_url:
        return False
    return any(predicted_url == normalize_text(target) for target in reference.all_urls())


def _matches_reference_item(predicted: dict[str, Any], reference: PrecisionMinerReferenceItem) -> bool:
    return _best_name_match(predicted, reference) or _best_url_match(predicted, reference)


def _explanation_hit(predicted: dict[str, Any], reference: PrecisionMinerReferenceItem) -> float | None:
    required_terms = [normalize_text(item) for item in reference.expected_explanation_contains if normalize_text(item)]
    if not required_terms:
        return None
    explanation = normalize_text(predicted.get("explanation", ""))
    if not explanation:
        return 0.0
    hits = sum(term in explanation for term in required_terms)
    return hits / len(required_terms)


def _score_items(
    predicted_items: list[dict[str, Any]],
    reference_items: list[PrecisionMinerReferenceItem],
) -> dict[str, float | None]:
    if not reference_items:
        return {
            "item_precision": None,
            "item_recall": None,
            "item_f1": None,
            "required_item_recall": None,
            "explanation_coverage": None,
        }

    matched_reference_indexes: set[int] = set()
    matched_predicted_indexes: set[int] = set()
    explanation_scores: list[float] = []

    for pred_idx, predicted in enumerate(predicted_items):
        for ref_idx, reference in enumerate(reference_items):
            if ref_idx in matched_reference_indexes:
                continue
            if not _matches_reference_item(predicted, reference):
                continue
            matched_reference_indexes.add(ref_idx)
            matched_predicted_indexes.add(pred_idx)
            explanation_score = _explanation_hit(predicted, reference)
            if explanation_score is not None:
                explanation_scores.append(explanation_score)
            break

    precision = len(matched_predicted_indexes) / len(predicted_items) if predicted_items else 0.0
    recall = len(matched_reference_indexes) / len(reference_items) if reference_items else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    required_indexes = {idx for idx, item in enumerate(reference_items) if item.required}
    if required_indexes:
        required_hits = len(required_indexes & matched_reference_indexes)
        required_recall = required_hits / len(required_indexes)
    else:
        required_recall = None

    explanation_coverage = (
        sum(explanation_scores) / len(explanation_scores) if explanation_scores else None
    )

    return {
        "item_precision": precision,
        "item_recall": recall,
        "item_f1": f1,
        "required_item_recall": required_recall,
        "explanation_coverage": explanation_coverage,
    }


def run_precision_miner_case(
    case: PrecisionMinerEvalCase,
    config: RagPipelineConfig,
) -> PrecisionMinerCaseRunResult:
    llm_provider = _enum(LLMProvider, config.llm_provider)
    tempdir = None

    try:
        if case.paper_path is not None:
            paper_path = Path(case.paper_path)
            if not paper_path.is_file():
                raise ValueError(
                    "Precision miner eval currently expects each case paper_path to point to a single file."
                )
            paper = _load_paper(
                paper_path,
                loader_kind=_enum(LoaderKind, config.loader),
                paper_id=case.paper_id,
            )
            tempdir, papers, retriever = _transient_retriever_for_path(
                paper_path,
                loader_kind=_enum(LoaderKind, config.loader),
                embed_model=config.embed_model,
                chunker_kind=_enum(ChunkerKind, config.chunker),
                min_chunk_size=config.min_chunk_size,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if len(papers) != 1:
                raise ValueError("Precision miner eval transient mode expects a single loaded paper.")
            paper_id = paper.paper_id
            metadata = paper.metadata
        else:
            raise ValueError("Precision miner eval currently requires paper_path for each case.")

        generator = _build_generator(llm_provider, config.llm_model, config.temperature)
        miner = PrecisionMiner(
            retriever=retriever,
            generator=generator,
            config=_build_precision_miner_config(
                _enum(PrecisionMinerKind, case.miner_kind),
                config.top_k,
            ),
            academic_db=None,
        )
        detailed = miner.run_detailed(paper_id, metadata=metadata)
        predicted_items = [item.model_dump() for item in detailed.result.items]
        retrieved_contexts = [chunk.text for chunk in detailed.relevant_chunks]

        all_reference_contexts = [item.text for item in case.reference_contexts]
        must_retrieve_contexts = [item.text for item in case.reference_contexts if item.must_retrieve]
        support_precision, support_recall, support_f1 = reference_context_prf(
            retrieved_contexts,
            all_reference_contexts,
            threshold=config.local_context_match_threshold,
        )
        _, must_retrieve_recall, _ = reference_context_prf(
            retrieved_contexts,
            must_retrieve_contexts,
            threshold=config.local_context_match_threshold,
        )
        item_scores = _score_items(predicted_items, case.reference_items)

        return PrecisionMinerCaseRunResult(
            case_id=case.case_id,
            miner_kind=case.miner_kind,
            paper_path=case.paper_path,
            paper_id=paper_id,
            reference_answer=case.reference_answer,
            response_description=detailed.result.description,
            reference_items=[asdict(item) for item in case.reference_items],
            predicted_items=predicted_items,
            reference_contexts=[asdict(item) for item in case.reference_contexts],
            retrieved_contexts=retrieved_contexts,
            retrieved_chunks=[
                {
                    "id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "text": chunk.text,
                    "section_type": chunk.section_type,
                    "section_title": chunk.section_title,
                    "is_metadata": chunk.is_metadata,
                    "source": chunk.source,
                    "similarity_score": chunk.similarity_score,
                    "rank_score": chunk.rank_score,
                    "artifacts": dict(chunk.artifacts),
                }
                for chunk in detailed.relevant_chunks
            ],
            retrieval_count=len(detailed.relevant_chunks),
            tags=list(case.tags),
            notes=case.notes,
            support_context_precision=support_precision,
            support_context_recall=support_recall,
            support_context_f1=support_f1,
            must_retrieve_context_recall=must_retrieve_recall,
            description_exact_match=float(
                normalize_text(detailed.result.description) == normalize_text(case.reference_answer)
            ),
            **item_scores,
        )
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def run_precision_miner_cases(
    cases: list[PrecisionMinerEvalCase],
    config: RagPipelineConfig,
) -> list[PrecisionMinerCaseRunResult]:
    results: list[PrecisionMinerCaseRunResult] = []
    for case in cases:
        try:
            results.append(run_precision_miner_case(case, config))
        except Exception as exc:
            if not config.continue_on_error:
                raise
            results.append(
                PrecisionMinerCaseRunResult(
                    case_id=case.case_id,
                    miner_kind=case.miner_kind,
                    paper_path=case.paper_path,
                    paper_id=case.paper_id,
                    reference_answer=case.reference_answer,
                    response_description=None,
                    reference_items=[asdict(item) for item in case.reference_items],
                    reference_contexts=[asdict(item) for item in case.reference_contexts],
                    tags=list(case.tags),
                    notes=case.notes,
                    run_error=str(exc),
                )
            )
    return results


def summarize_precision_miner_runs(runs: list[PrecisionMinerCaseRunResult]) -> dict[str, Any]:
    successful_runs = [run for run in runs if run.run_error is None]
    summary: dict[str, Any] = {
        "row_count": len(runs),
        "successful_runs": len(successful_runs),
        "failed_runs": len(runs) - len(successful_runs),
    }
    for metric_name in (
        "item_precision",
        "item_recall",
        "item_f1",
        "required_item_recall",
        "support_context_precision",
        "support_context_recall",
        "support_context_f1",
        "must_retrieve_context_recall",
        "explanation_coverage",
        "description_exact_match",
    ):
        values = [getattr(run, metric_name) for run in successful_runs if getattr(run, metric_name) is not None]
        if values:
            series = pd.Series(values)
            summary[metric_name] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return summary


def summarize_precision_miner_scores(scores: pd.DataFrame, metric_names: list[str]) -> dict[str, Any]:
    return summarize_ragas_scores(scores, metric_names)
