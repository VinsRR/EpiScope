from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from eval.common.io import read_ground_truth
from eval.common.labels import parse_label_set
from eval.common.paths import discover_result_files, parse_result_path
from episcope.episcope import (
    IndexBackend,
    LLMProvider,
    PrecisionMinerKind,
    RetrievalMode,
    _build_generator,
    _build_precision_miner_config,
    _build_retriever,
    _build_vector_db,
)
from episcope.schemas import PaperMetadata
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
    paper_id: Optional[str] = None
    reference_contexts: list[PrecisionMinerReferenceContext] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
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
    expected_present: Optional[bool] = None
    predicted_present: Optional[bool] = None
    presence_agreement: Optional[float] = None
    url_precision: Optional[float] = None
    url_recall: Optional[float] = None
    url_f1: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DATA_SOURCE_LABELS = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "OPEN",
    "REPORTED",
    "REFERENCED",
    "AVAILABLE_UPON_REQUEST",
    "UPON_REQUEST",
    "CLOSED",
}
_URL_RE = re.compile(r"https?://[^\s\])}>\"']+", flags=re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", flags=re.IGNORECASE)
_SUPPLEMENTARY_RE = re.compile(
    r"\b("
    r"supplement(?:ary|al)?|supporting information|appendix|appendices|"
    r"additional file|additional material|S\d+\s+(?:table|fig(?:ure)?|file|data|text)"
    r")\b",
    flags=re.IGNORECASE,
)
_SOURCE_NAME_RE = re.compile(
    r"\b("
    r"World Health Organization|WHO|Centers for Disease Control and Prevention|CDC|"
    r"Ministry of Health|Johns Hopkins|GenBank|GISAID|DDBJ|EMBL|"
    r"Public Health Reports|national vital statistics|vital statistics|"
    r"case report forms?|electronic medical records?|surveillance system|"
    r"contact tracing database|outbreak investigation reports?"
    r")\b",
    flags=re.IGNORECASE,
)
_SUPPLEMENTARY_NAME_RE = re.compile(
    r"\b("
    r"Supplementary\s+(?:Table|Figure|Fig\.?|File|Data|Material|Materials|Information|Text)\s*[A-Z0-9.-]*|"
    r"S\d+\s+(?:Table|Fig(?:ure)?|File|Data|Text)|"
    r"Appendix\s+[A-Z0-9.-]+|"
    r"Supporting Information"
    r")\b",
    flags=re.IGNORECASE,
)


def _compact_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique_strings(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = _compact_whitespace(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _parse_jsonish(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _texts_from_payload(value: Any) -> list[str]:
    parsed = _parse_jsonish(value)
    texts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("text", "raw_text", "content", "evidence", "answer"):
                candidate = item.get(key)
                if isinstance(candidate, str):
                    texts.append(candidate)
            for nested in item.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(parsed)
    return texts


def _extract_urls(texts: Iterable[str]) -> list[str]:
    urls: list[str] = []
    for text in texts:
        urls.extend(match.rstrip(".,;") for match in _URL_RE.findall(text))
        urls.extend(f"https://doi.org/{match.rstrip('.,;')}" for match in _DOI_RE.findall(text))
    return _unique_strings(urls)


def _guess_data_source_names(texts: Iterable[str]) -> list[str]:
    names: list[str] = []
    for text in texts:
        names.extend(match.group(1) for match in _SOURCE_NAME_RE.finditer(text))
    return _unique_strings(names, limit=8)


def _guess_supplementary_names(texts: Iterable[str]) -> list[str]:
    names: list[str] = []
    for text in texts:
        names.extend(match.group(1) for match in _SUPPLEMENTARY_NAME_RE.finditer(text))
    return _unique_strings(names, limit=8)


def _row_reference_texts(row: pd.Series) -> list[str]:
    values = [row.get("evidence", ""), row.get("provenance_answer", "")]
    for column in (
        "top_evidence_json",
        "provenance_evidences_json",
        "trace_raw_llm_response",
        "extras",
    ):
        if column in row:
            values.extend(_texts_from_payload(row.get(column)))
    return _unique_strings(str(value) for value in values if _compact_whitespace(value))


def load_sampled_papers(path: str | Path) -> pd.DataFrame:
    sampled_path = Path(path)
    if not sampled_path.exists():
        raise FileNotFoundError(f"Sampled papers file not found: {sampled_path}")
    df = read_ground_truth(sampled_path)
    if "paper_id" not in df.columns:
        raise ValueError(f"Sampled papers file {sampled_path} is missing required column 'paper_id'.")
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    empty = df[df["paper_id"] == ""]
    if not empty.empty:
        raise ValueError(f"Sampled papers file {sampled_path} contains empty paper_id values.")
    return df.drop_duplicates(subset=["paper_id"], keep="first").reset_index(drop=True)


def discover_data_accessibility_tsvs(
    run_roots: Iterable[str | Path],
    *,
    pattern: str = "final_*.tsv",
) -> list[Path]:
    files: list[Path] = []
    for path in discover_result_files(run_roots, pattern=pattern):
        try:
            meta = parse_result_path(path)
        except ValueError:
            continue
        if meta["task"] == "data-accessibility":
            files.append(path)
    return sorted(files)


def read_data_accessibility_tsv(path: str | Path) -> pd.DataFrame:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Data-accessibility TSV not found: {result_path}")
    try:
        df = pd.read_csv(result_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not read data-accessibility TSV {result_path}: {exc}") from exc

    required = {"paper_id", "classification", "evidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {result_path}: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    malformed = df[df["paper_id"] == ""]
    if not malformed.empty:
        raise ValueError(f"Data-accessibility TSV {result_path} contains rows with empty paper_id.")
    df["source_file"] = str(result_path)
    return df


def load_data_accessibility_outputs(
    run_roots: Iterable[str | Path],
    *,
    pattern: str = "final_*.tsv",
) -> pd.DataFrame:
    files = discover_data_accessibility_tsvs(run_roots, pattern=pattern)
    if not files:
        roots = ", ".join(str(Path(root)) for root in run_roots)
        raise FileNotFoundError(
            f"No data-accessibility result TSVs matching {pattern!r} were found under: {roots}"
        )
    frames = [read_data_accessibility_tsv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _metadata_from_sampled_row(row: pd.Series) -> dict[str, Any]:
    year = row.get("year")
    try:
        publication_year = int(float(year)) if _compact_whitespace(year) else None
    except ValueError:
        publication_year = None
    return {
        "title": _compact_whitespace(row.get("title")),
        "journal": _compact_whitespace(row.get("journal")),
        "publication_year": publication_year,
        "doi": _compact_whitespace(row.get("paper_id")),
        "file_path": _compact_whitespace(row.get("filepath")),
    }


def _reference_item(
    *,
    canonical_name: str,
    aliases: list[str],
    url: str | None,
    reference_kind: str,
    expected_present: bool,
) -> dict[str, Any]:
    return {
        "canonical_name": canonical_name,
        "aliases": aliases,
        "url": url,
        "url_aliases": [],
        "required": expected_present,
        "reference_kind": reference_kind,
        "expected_present": expected_present,
    }


def build_precision_miner_references(
    *,
    sampled_papers_path: str | Path,
    data_accessibility_roots: Iterable[str | Path],
    pattern: str = "final_*.tsv",
    max_contexts_per_paper: int = 8,
) -> pd.DataFrame:
    sampled = load_sampled_papers(sampled_papers_path)
    outputs = load_data_accessibility_outputs(data_accessibility_roots, pattern=pattern)

    sampled_ids = set(sampled["paper_id"])
    outputs = outputs[outputs["paper_id"].isin(sampled_ids)].copy()
    available_ids = set(outputs["paper_id"])
    missing_ids = sorted(sampled_ids - available_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        raise ValueError(
            f"Data-accessibility outputs are missing {len(missing_ids)} sampled paper_id(s): "
            f"{preview}{suffix}"
        )

    sampled_by_id = sampled.set_index("paper_id", drop=False)
    rows: list[dict[str, Any]] = []
    for paper_id, group in outputs.groupby("paper_id", sort=True):
        labels = sorted(
            set().union(*(parse_label_set(value) for value in group["classification"]))
        )
        reference_texts = _unique_strings(
            text
            for _, row in group.iterrows()
            for text in _row_reference_texts(row)
            if _compact_whitespace(text)
        )
        contexts = reference_texts[:max_contexts_per_paper]
        supplementary_texts = [
            text for text in reference_texts if _SUPPLEMENTARY_RE.search(text)
        ]
        expected_data_source = bool(set(labels) & _DATA_SOURCE_LABELS)
        expected_supplementary_link = bool(supplementary_texts)

        source_names = _guess_data_source_names(reference_texts)
        source_urls = _extract_urls(reference_texts)
        data_source_items = []
        if expected_data_source:
            names = source_names or ["Data source evidence from data-accessibility outputs"]
            data_source_items = [
                _reference_item(
                    canonical_name=name,
                    aliases=[],
                    url=source_urls[0] if source_urls else None,
                    reference_kind="DataSource",
                    expected_present=True,
                )
                for name in names
            ]

        supplementary_names = _guess_supplementary_names(supplementary_texts)
        supplementary_urls = _extract_urls(supplementary_texts)
        supplementary_items = []
        if expected_supplementary_link:
            names = supplementary_names or ["Supplementary material"]
            supplementary_items = [
                _reference_item(
                    canonical_name=name,
                    aliases=[],
                    url=supplementary_urls[0] if supplementary_urls else None,
                    reference_kind="SupplementaryLink",
                    expected_present=True,
                )
                for name in names
            ]

        sampled_row = sampled_by_id.loc[paper_id]
        metadata = _metadata_from_sampled_row(sampled_row)
        rows.append(
            {
                "paper_id": paper_id,
                "title": metadata["title"],
                "journal": metadata["journal"],
                "year": metadata["publication_year"],
                "filepath": metadata["file_path"],
                "availability_labels": ",".join(labels),
                "expected_data_source": expected_data_source,
                "expected_supplementary_link": expected_supplementary_link,
                "data_source_reference": "\n\n".join(contexts),
                "supplementary_link_reference": "\n\n".join(supplementary_texts[:max_contexts_per_paper]),
                "reference_contexts_json": json.dumps(contexts, ensure_ascii=False),
                "data_source_reference_items_json": json.dumps(data_source_items, ensure_ascii=False),
                "supplementary_link_reference_items_json": json.dumps(supplementary_items, ensure_ascii=False),
                "source_files": json.dumps(sorted(group["source_file"].unique()), ensure_ascii=False),
                "source_row_count": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values("paper_id").reset_index(drop=True)


def _case_metadata_from_reference_row(row: pd.Series, *, reference_kind: str, expected_present: bool) -> dict[str, Any]:
    year = row.get("year")
    try:
        publication_year = int(year) if _compact_whitespace(year) else None
    except (TypeError, ValueError):
        publication_year = None
    return {
        "title": row.get("title") or "",
        "journal": row.get("journal") or "",
        "publication_year": publication_year,
        "doi": row.get("paper_id") or "",
        "file_path": row.get("filepath") or "",
        "reference_kind": reference_kind,
        "expected_present": bool(expected_present),
        "availability_labels": row.get("availability_labels") or "",
    }


def build_precision_miner_cases_from_references(
    references: pd.DataFrame,
) -> list[PrecisionMinerEvalCase]:
    cases: list[PrecisionMinerEvalCase] = []
    for row in references.to_dict(orient="records"):
        contexts = [
            PrecisionMinerReferenceContext(text=text, must_retrieve=True)
            for text in (_parse_jsonish(row.get("reference_contexts_json")) or [])
            if _compact_whitespace(text)
        ]

        for reference_kind, miner_kind, expected_column, answer_column, items_column in (
            (
                "DataSource",
                "find_data_sources",
                "expected_data_source",
                "data_source_reference",
                "data_source_reference_items_json",
            ),
            (
                "SupplementaryLink",
                "find_supplementary_links",
                "expected_supplementary_link",
                "supplementary_link_reference",
                "supplementary_link_reference_items_json",
            ),
        ):
            expected_present = bool(row.get(expected_column))
            item_records = _parse_jsonish(row.get(items_column)) or []
            reference_items = [
                PrecisionMinerReferenceItem(
                    canonical_name=str(item.get("canonical_name") or item.get("name") or reference_kind),
                    aliases=list(item.get("aliases", []) or []),
                    url=item.get("url"),
                    url_aliases=list(item.get("url_aliases", []) or []),
                    required=bool(item.get("required", expected_present)),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"canonical_name", "name", "aliases", "url", "url_aliases", "required"}
                    },
                )
                for item in item_records
            ]
            cases.append(
                PrecisionMinerEvalCase(
                    schema_version="1",
                    case_id=f"{row['paper_id']}::{reference_kind}",
                    miner_kind=miner_kind,
                    paper_id=str(row["paper_id"]),
                    reference_answer=_compact_whitespace(row.get(answer_column))
                    or (
                        f"{reference_kind} expected."
                        if expected_present
                        else f"No {reference_kind} reference inferred."
                    ),
                    reference_items=reference_items,
                    reference_contexts=contexts if expected_present else [],
                    tags=["sampled_papers_full", "auto_reference", reference_kind],
                    metadata=_case_metadata_from_reference_row(
                        pd.Series(row),
                        reference_kind=reference_kind,
                        expected_present=expected_present,
                    ),
                    notes="Reference constructed automatically from data-accessibility outputs.",
                )
            )
    return cases


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


def _metadata_from_case(case: PrecisionMinerEvalCase) -> PaperMetadata:
    metadata = dict(case.metadata or {})
    return PaperMetadata(
        title=str(metadata.get("title") or case.paper_id or case.case_id),
        abstract=str(metadata.get("abstract") or ""),
        journal=str(metadata.get("journal") or ""),
        publication_year=metadata.get("publication_year"),
        doi=str(metadata.get("doi") or case.paper_id or ""),
        keywords=list(metadata.get("keywords", []) or []),
        file_path=metadata.get("file_path"),
    )


def _score_urls(
    predicted_items: list[dict[str, Any]],
    reference_items: list[PrecisionMinerReferenceItem],
) -> dict[str, float | None]:
    predicted_urls = {
        normalize_text(url)
        for item in predicted_items
        for url in [item.get("url")]
        if normalize_text(url or "")
    }
    reference_urls = {
        normalize_text(url)
        for reference in reference_items
        for url in reference.all_urls()
        if normalize_text(url)
    }
    if not reference_urls and not predicted_urls:
        return {"url_precision": None, "url_recall": None, "url_f1": None}
    hits = len(predicted_urls & reference_urls)
    precision = hits / len(predicted_urls) if predicted_urls else 0.0
    recall = hits / len(reference_urls) if reference_urls else None
    if recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"url_precision": precision, "url_recall": recall, "url_f1": f1}


def run_precision_miner_case(
    case: PrecisionMinerEvalCase,
    config: RagPipelineConfig,
) -> PrecisionMinerCaseRunResult:
    llm_provider = _enum(LLMProvider, config.llm_provider)

    if case.paper_id is None:
        raise ValueError("Precision-miner eval cases must include paper_id.")

    vectordb = _build_vector_db(
        _enum(IndexBackend, config.index_backend),
        index_dir=Path(config.index_dir),
        qdrant_collection=config.qdrant_collection,
        qdrant_url=config.qdrant_url,
    )
    if not list(vectordb.get_points(namespace=case.paper_id)):
        raise ValueError(
            f"No indexed chunks were found for sampled paper_id={case.paper_id!r}. "
            "Check the vector DB index backend, index directory/collection, and sampled_papers_full.csv."
        )
    retriever = _build_retriever(
        vectordb,
        retrieval_mode=_enum(RetrievalMode, config.retrieval_mode),
    )

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
    detailed = miner.run_detailed(case.paper_id, metadata=_metadata_from_case(case))
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
    url_scores = _score_urls(predicted_items, case.reference_items)
    expected_present = bool(case.metadata.get("expected_present", bool(case.reference_items)))
    predicted_present = bool(predicted_items)

    return PrecisionMinerCaseRunResult(
        case_id=case.case_id,
        miner_kind=case.miner_kind,
        paper_id=case.paper_id,
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
        expected_present=expected_present,
        predicted_present=predicted_present,
        presence_agreement=float(expected_present == predicted_present),
        **item_scores,
        **url_scores,
    )


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
                    paper_id=case.paper_id,
                    reference_answer=case.reference_answer,
                    response_description=None,
                    reference_items=[asdict(item) for item in case.reference_items],
                    reference_contexts=[asdict(item) for item in case.reference_contexts],
                    tags=list(case.tags),
                    notes=case.notes,
                    run_error=str(exc),
                    expected_present=bool(case.metadata.get("expected_present", bool(case.reference_items))),
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
        "presence_agreement",
        "url_precision",
        "url_recall",
        "url_f1",
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


def precision_miner_agreement_metrics(
    runs: list[PrecisionMinerCaseRunResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, list[PrecisionMinerCaseRunResult]]] = [
        ("all", runs),
        *[
            (miner_kind, [run for run in runs if run.miner_kind == miner_kind])
            for miner_kind in sorted({run.miner_kind for run in runs})
        ],
    ]
    for miner_kind, group in groups:
        successful = [run for run in group if run.run_error is None]
        expected_positive = [
            run for run in successful if run.expected_present is True
        ]
        predicted_positive = [
            run for run in successful if run.predicted_present is True
        ]
        agreements = [
            run.presence_agreement
            for run in successful
            if run.presence_agreement is not None
        ]
        rows.append(
            {
                "miner_kind": miner_kind,
                "case_count": len(group),
                "successful_runs": len(successful),
                "failed_runs": len(group) - len(successful),
                "expected_positive": len(expected_positive),
                "predicted_positive": len(predicted_positive),
                "presence_accuracy": (
                    sum(agreements) / len(agreements) if agreements else None
                ),
                "mean_item_precision": _mean_metric(successful, "item_precision"),
                "mean_item_recall": _mean_metric(successful, "item_recall"),
                "mean_item_f1": _mean_metric(successful, "item_f1"),
                "mean_support_context_recall": _mean_metric(
                    successful,
                    "support_context_recall",
                ),
                "mean_url_recall": _mean_metric(successful, "url_recall"),
            }
        )
    return pd.DataFrame(rows)


def _mean_metric(runs: list[PrecisionMinerCaseRunResult], metric_name: str) -> float | None:
    values = [getattr(run, metric_name) for run in runs if getattr(run, metric_name) is not None]
    if not values:
        return None
    return float(pd.Series(values).mean())


def summarize_precision_miner_scores(scores: pd.DataFrame, metric_names: list[str]) -> dict[str, Any]:
    return summarize_ragas_scores(scores, metric_names)
