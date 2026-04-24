from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from episcope.rag.provenance import Evidence, Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.output import (
    ClassificationDecision,
    ClassificationTrace,
    ClassificationTrainingRecord,
    CompletionSample,
)
from episcope.workflows.classification.schemas import ClassificationResult


class TaskBucket(str, Enum):
    JSON_VALIDITY = "json_validity"
    CLASSIFICATION_CORE = "classification_core"
    RL_JSON_VALIDITY_CANDIDATE = "rl_json_validity_candidate"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class CapturedTraceRecord:
    record_id: str
    created_at: str
    classifier_kind: str
    paper_id: str
    strategy_name: Optional[str]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    gold_label: Optional[Any] = None

    decision: Optional[ClassificationDecision] = None
    provenance: Optional[Provenance] = None
    trace: Optional[ClassificationTrace] = None
    training: Optional[ClassificationTrainingRecord] = None

    capture_version: str = "v1"


@dataclass
class ReviewDecision:
    record_id: str
    reviewed_at: str
    reviewer: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING

    buckets: List[TaskBucket] = field(default_factory=list)
    notes: str = ""

    edited_completion: Optional[str] = None
    updated_label: Optional[List[str]] = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def _serialize_search_result(chunk: SearchResult) -> Dict[str, Any]:
    return {
        "id": chunk.id,
        "paper_id": chunk.paper_id,
        "text": chunk.text,
        "section_type": chunk.section_type,
        "title": chunk.title,
        "similarity_score": chunk.similarity_score,
        "source": chunk.source,
        "artifacts": _json_ready(chunk.artifacts),
        "rank_score": chunk.rank_score,
    }


def _deserialize_search_result(data: Dict[str, Any]) -> SearchResult:
    return SearchResult(
        id=data.get("id", ""),
        paper_id=data.get("paper_id", ""),
        text=data.get("text", ""),
        section_type=data.get("section_type", "other"),
        title=data.get("title", ""),
        similarity_score=data.get("similarity_score", 0.0),
        source=data.get("source", ""),
        artifacts=data.get("artifacts", {}) or {},
        rank_score=data.get("rank_score", 0.0),
    )


def _serialize_classification_result(result: ClassificationResult) -> Dict[str, Any]:
    return {
        "classification": _json_ready(result.classification),
        "confidence": result.confidence,
        "class_probabilities": _json_ready(result.class_probabilities),
        "evidence": _json_ready(result.evidence),
        "extras": _json_ready(result.extras),
    }


def _deserialize_classification_result(data: Dict[str, Any]) -> ClassificationResult:
    return ClassificationResult(
        classification=list(data.get("classification", []) or []),
        confidence=float(data.get("confidence", 0.0)),
        class_probabilities=dict(data.get("class_probabilities", {}) or {}),
        evidence=dict(data.get("evidence", {}) or {}),
        extras=dict(data.get("extras", {}) or {}),
    )


def _serialize_decision(
    decision: Optional[ClassificationDecision],
) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    return {
        "paper_id": decision.paper_id,
        "metadata": decision.metadata.to_dict(),
        "result": _serialize_classification_result(decision.result),
        "top_evidence": [
            _serialize_search_result(chunk) for chunk in decision.top_evidence
        ],
    }


def _deserialize_decision(
    data: Optional[Dict[str, Any]],
) -> Optional[ClassificationDecision]:
    if data is None:
        return None
    return ClassificationDecision(
        paper_id=data.get("paper_id", ""),
        metadata=PaperMetadata.from_dict(data.get("metadata", {}) or {}),
        result=_deserialize_classification_result(data.get("result", {}) or {}),
        top_evidence=[
            _deserialize_search_result(item)
            for item in data.get("top_evidence", []) or []
        ],
    )


def _serialize_evidence(item: Evidence) -> Dict[str, Any]:
    return {
        "paper_id": item.paper_id,
        "snippet": item.snippet,
        "section": item.section,
        "index_version": item.index_version,
        "model_id": item.model_id,
        "prompt_id": item.prompt_id,
    }


def _deserialize_evidence(data: Dict[str, Any]) -> Evidence:
    return Evidence(
        paper_id=data.get("paper_id"),
        snippet=data.get("snippet"),
        section=data.get("section"),
        index_version=data.get("index_version"),
        model_id=data.get("model_id"),
        prompt_id=data.get("prompt_id"),
    )


def _serialize_provenance(provenance: Optional[Provenance]) -> Optional[Dict[str, Any]]:
    if provenance is None:
        return None
    return {
        "answer": provenance.answer,
        "evidences": [_serialize_evidence(item) for item in provenance.evidences],
    }


def _deserialize_provenance(data: Optional[Dict[str, Any]]) -> Optional[Provenance]:
    if data is None:
        return None
    return Provenance(
        answer=data.get("answer", ""),
        evidences=[
            _deserialize_evidence(item) for item in data.get("evidences", []) or []
        ],
    )


def _serialize_trace(trace: Optional[ClassificationTrace]) -> Optional[Dict[str, Any]]:
    if trace is None:
        return None
    return {
        "prompt_messages": _json_ready(trace.prompt_messages),
        "raw_llm_response": trace.raw_llm_response,
    }


def _deserialize_trace(data: Optional[Dict[str, Any]]) -> Optional[ClassificationTrace]:
    if data is None:
        return None
    return ClassificationTrace(
        prompt_messages=list(data.get("prompt_messages", []) or []),
        raw_llm_response=data.get("raw_llm_response"),
    )


def _serialize_completion_sample(sample: CompletionSample) -> Dict[str, Any]:
    return {
        "messages": _json_ready(sample.messages),
        "completion": sample.completion,
        "parsed_ok": sample.parsed_ok,
        "reward": sample.reward,
    }


def _deserialize_completion_sample(data: Dict[str, Any]) -> CompletionSample:
    return CompletionSample(
        messages=list(data.get("messages", []) or []),
        completion=data.get("completion", ""),
        parsed_ok=bool(data.get("parsed_ok", False)),
        reward=data.get("reward"),
    )


def _serialize_training(
    training: Optional[ClassificationTrainingRecord],
) -> Optional[Dict[str, Any]]:
    if training is None:
        return None
    return {
        "paper_id": training.paper_id,
        "result": _serialize_classification_result(training.result),
        "prompt_messages": _json_ready(training.prompt_messages),
        "raw_llm_response": training.raw_llm_response,
        "all_samples": [
            _serialize_completion_sample(sample) for sample in training.all_samples
        ],
        "gold_label": _json_ready(training.gold_label),
    }


def _deserialize_training(
    data: Optional[Dict[str, Any]],
) -> Optional[ClassificationTrainingRecord]:
    if data is None:
        return None
    return ClassificationTrainingRecord(
        paper_id=data.get("paper_id", ""),
        result=_deserialize_classification_result(data.get("result", {}) or {}),
        prompt_messages=list(data.get("prompt_messages", []) or []),
        raw_llm_response=data.get("raw_llm_response"),
        all_samples=[
            _deserialize_completion_sample(item)
            for item in data.get("all_samples", []) or []
        ],
        gold_label=data.get("gold_label"),
    )


def serialize_captured_trace_record(record: CapturedTraceRecord) -> Dict[str, Any]:
    return {
        "record_id": record.record_id,
        "created_at": record.created_at,
        "classifier_kind": record.classifier_kind,
        "paper_id": record.paper_id,
        "strategy_name": record.strategy_name,
        "config_snapshot": _json_ready(record.config_snapshot),
        "gold_label": _json_ready(record.gold_label),
        "decision": _serialize_decision(record.decision),
        "provenance": _serialize_provenance(record.provenance),
        "trace": _serialize_trace(record.trace),
        "training": _serialize_training(record.training),
        "capture_version": record.capture_version,
    }


def deserialize_captured_trace_record(data: Dict[str, Any]) -> CapturedTraceRecord:
    return CapturedTraceRecord(
        record_id=data["record_id"],
        created_at=data["created_at"],
        classifier_kind=data["classifier_kind"],
        paper_id=data["paper_id"],
        strategy_name=data.get("strategy_name"),
        config_snapshot=dict(data.get("config_snapshot", {}) or {}),
        gold_label=data.get("gold_label"),
        decision=_deserialize_decision(data.get("decision")),
        provenance=_deserialize_provenance(data.get("provenance")),
        trace=_deserialize_trace(data.get("trace")),
        training=_deserialize_training(data.get("training")),
        capture_version=data.get("capture_version", "v1"),
    )


def serialize_review_decision(review: ReviewDecision) -> Dict[str, Any]:
    return {
        "record_id": review.record_id,
        "reviewed_at": review.reviewed_at,
        "reviewer": review.reviewer,
        "status": review.status.value,
        "buckets": [bucket.value for bucket in review.buckets],
        "notes": review.notes,
        "edited_completion": review.edited_completion,
        "updated_label": list(review.updated_label or []),
    }


def deserialize_review_decision(data: Dict[str, Any]) -> ReviewDecision:
    return ReviewDecision(
        record_id=data["record_id"],
        reviewed_at=data["reviewed_at"],
        reviewer=data.get("reviewer"),
        status=ReviewStatus(data.get("status", ReviewStatus.PENDING.value)),
        buckets=[TaskBucket(value) for value in data.get("buckets", []) or []],
        notes=data.get("notes", ""),
        edited_completion=data.get("edited_completion"),
        updated_label=list(data.get("updated_label", []) or []) or None,
    )
