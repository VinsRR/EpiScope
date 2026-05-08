from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from episcope.schemas import PaperMetadata
from episcope.workflows.classification.config import (
    BaseClassifierConfig,
    DataAccessibilityClassifierConfig,
    DataTypeClassifierConfig,
    GeoClassifierConfig,
    PaperTypeClassifierConfig,
)
from episcope.workflows.classification.schemas import (
    ClassificationResult,
    DataAccessibility,
    DataType,
    GeoRegion,
    PaperType,
)

CLASSIFIER_CONFIGS: dict[str, type[BaseClassifierConfig]] = {
    "paper_type": PaperTypeClassifierConfig,
    "data_accessibility": DataAccessibilityClassifierConfig,
    "data_type": DataTypeClassifierConfig,
    "geo": GeoClassifierConfig,
}

GT_COLUMNS: dict[str, str] = {
    "paper_type": "ptype_classification",
    "data_accessibility": "availability_classification",
    "data_type": "data_type_classification",
    "geo": "geo_classification",
}

TASK_SLUGS: dict[str, str] = {
    "paper_type": "paper-type",
    "data_accessibility": "data-accessibility",
    "data_type": "data-type",
    "geo": "geo",
}

ENUMS_BY_KIND: dict[str, type[Any]] = {
    "paper_type": PaperType,
    "data_accessibility": DataAccessibility,
    "data_type": DataType,
    "geo": GeoRegion,
}

_CURLY_TO_STRAIGHT = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)


@dataclass(frozen=True)
class BaselinePrediction:
    """Small wrapper for a baseline output plus optional trace text."""

    result: ClassificationResult
    raw_response: str | None = None
    prompt_messages: list[dict[str, str]] | None = None


def classifier_config(classifier_kind: str) -> BaseClassifierConfig:
    try:
        return CLASSIFIER_CONFIGS[classifier_kind]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown classifier_kind={classifier_kind!r}. "
            f"Use one of {sorted(CLASSIFIER_CONFIGS)}."
        ) from exc


def ground_truth_column(classifier_kind: str) -> str:
    try:
        return GT_COLUMNS[classifier_kind]
    except KeyError as exc:
        raise ValueError(
            f"No ground-truth column configured for classifier_kind={classifier_kind!r}."
        ) from exc


def task_slug(classifier_kind: str) -> str:
    return TASK_SLUGS.get(classifier_kind, classifier_kind.replace("_", "-"))


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def stable_hash(value: Mapping[str, Any]) -> str:
    import hashlib

    blob = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()[:12]


def canonicalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_CURLY_TO_STRAIGHT)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def parse_label_names(value: Any) -> tuple[str, ...]:
    """Parse benchmark or TSV label payloads into canonical enum-name strings."""

    if value is None:
        return ()
    try:
        import pandas as pd

        if pd.isna(value):
            return ()
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            sorted(
                {
                    canonicalize_text(getattr(item, "name", item)).upper()
                    for item in value
                    if canonicalize_text(getattr(item, "name", item))
                }
            )
        )

    text = canonicalize_text(value)
    if not text:
        return ()

    parsed: Any | None = None
    if (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

    if isinstance(parsed, (list, tuple, set, frozenset)):
        return parse_label_names(list(parsed))
    if isinstance(parsed, str):
        return parse_label_names(parsed)

    stripped = re.sub(r"^\[|\]$", "", text).strip()
    parts = stripped.split(",") if "," in stripped else [stripped]
    return tuple(
        sorted({canonicalize_text(part).strip("'\"").upper() for part in parts if part})
    )


def enum_label_names(classifier_kind: str) -> dict[str, Any]:
    enum_cls = ENUMS_BY_KIND[classifier_kind]
    out: dict[str, Any] = {}
    for item in enum_cls:
        out[item.name.upper()] = item
        out[str(item.value).upper()] = item
    return out


def labels_from_names(classifier_kind: str, names: Iterable[str]) -> list[Any]:
    mapping = enum_label_names(classifier_kind)
    labels = []
    for name in names:
        mapped = mapping.get(canonicalize_text(name).upper())
        if mapped is not None and mapped not in labels:
            labels.append(mapped)
    return labels


def label_from_category(classifier_kind: str, category: str) -> Any | None:
    """Map a classifier config category/template key to the corresponding enum."""

    config = classifier_config(classifier_kind)
    raw_category = canonicalize_text(category)
    key = raw_category.lower()

    mapped = config.classification_mapping.get(category)
    if mapped is None:
        mapped = config.classification_mapping.get(raw_category)
    if mapped is not None:
        return mapped

    enum_cls = ENUMS_BY_KIND[classifier_kind]
    for label in enum_cls:
        if label.name.lower() == key or str(label.value).lower() == key:
            return label

    if classifier_kind == "data_accessibility" and key == "upon_request":
        return DataAccessibility.AVAILABLE_UPON_REQUEST
    return None


def default_labels(classifier_kind: str) -> list[Any]:
    return list(classifier_config(classifier_kind).default_classification or [])


def result_from_labels(
    classifier_kind: str,
    labels: Sequence[Any],
    *,
    confidence: float,
    reasoning: str,
    class_probabilities: Mapping[str, float] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> ClassificationResult:
    clean = list(labels) or default_labels(classifier_kind)
    return ClassificationResult(
        classification=clean,
        confidence=confidence,
        class_probabilities=dict(class_probabilities or {}),
        evidence={"reasoning": reasoning},
        extras=dict(extras or {}),
    )


def majority_label_set(label_sets: Iterable[Sequence[str]]) -> tuple[str, ...]:
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(sorted(label_set)) for label_set in label_sets if label_set
    )
    if not counts:
        return ()
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def metadata_from_mapping(record: Mapping[str, Any]) -> PaperMetadata:
    keywords = record.get("keywords") or record.get("keyword") or []
    if isinstance(keywords, str):
        keywords = [
            item.strip()
            for item in re.split(r"[;,]", keywords)
            if item and item.strip()
        ]
    year = record.get("publication_year", record.get("year"))
    try:
        publication_year = int(float(year)) if year not in (None, "") else None
    except Exception:
        publication_year = None
    return PaperMetadata(
        title=str(record.get("title") or ""),
        abstract=str(record.get("abstract") or ""),
        journal=str(record.get("journal") or ""),
        publication_year=publication_year,
        doi=str(record.get("paper_id") or record.get("doi") or ""),
        keywords=list(keywords or []),
        file_path=record.get("filepath") or record.get("file_path"),
    )


def metadata_text(
    metadata: PaperMetadata, record: Mapping[str, Any] | None = None
) -> str:
    record = record or {}
    cached = canonicalize_text(record.get("_metadata_text"))
    if cached:
        return cached
    parts = [
        metadata.title,
        metadata.abstract,
        metadata.journal,
        " ".join(metadata.keywords or []),
        str(metadata.publication_year or ""),
    ]
    for key in ("mesh_terms", "subject", "source", "notes"):
        if record.get(key):
            parts.append(str(record[key]))
    return "\n".join(part for part in parts if canonicalize_text(part)).strip()


def enum_names(labels: Sequence[Any]) -> list[str]:
    return [str(getattr(label, "name", label)) for label in labels]


def result_to_tsv_row(
    paper_id: str,
    prediction: BaselinePrediction | ClassificationResult,
    *,
    classifier_kind: str,
    baseline_name: str,
) -> dict[str, Any]:
    if isinstance(prediction, BaselinePrediction):
        result = prediction.result
        raw_response = prediction.raw_response
        messages = prediction.prompt_messages or []
    else:
        result = prediction
        raw_response = None
        messages = []

    reasoning = (
        result.evidence.get("reasoning")
        if isinstance(result.evidence, dict)
        else result.evidence
    )
    row: dict[str, Any] = {
        "paper_id": str(paper_id),
        "classification": enum_names(result.classification),
        "class_probabilities": result.class_probabilities,
        "confidence": result.confidence,
        "evidence": reasoning,
        "extras": result.extras or {},
        "baseline_name": baseline_name,
        "trace_raw_llm_response": raw_response,
        "trace_prompt_message_count": len(messages),
        "trace_prompt_messages_json": json.dumps(messages, ensure_ascii=False),
        "llm_call_count": 1 if raw_response else 0,
    }
    if classifier_kind == "paper_type":
        row["secondary_labels"] = enum_names(result.extras.get("secondary_labels", []))
    if classifier_kind == "geo":
        row["countries"] = result.extras.get("countries", [])
        row["cities"] = result.extras.get("cities", [])
    return row
