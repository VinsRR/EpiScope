from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import GeneratedQueryReviewRecord, RagCaseRunResult, SimpleRagQaCase


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False))
            handle.write("\n")
    return output_path


def load_simple_rag_qa_cases(path: str | Path) -> list[SimpleRagQaCase]:
    dataset_path = Path(path).resolve()
    cases: list[SimpleRagQaCase] = []
    for record in read_jsonl(dataset_path):
        missing = {
            key
            for key in ("schema_version", "case_id", "user_input", "reference")
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

        cases.append(
            SimpleRagQaCase(
                schema_version=str(record["schema_version"]),
                case_id=str(record["case_id"]),
                paper_path=paper_path,
                paper_id=record.get("paper_id"),
                user_input=str(record["user_input"]),
                reference=str(record["reference"]),
                reference_contexts=list(record.get("reference_contexts", []) or []),
                tags=list(record.get("tags", []) or []),
                metadata=dict(record.get("metadata", {}) or {}),
                rubrics=dict(record.get("rubrics", {}) or {}),
                notes=str(record.get("notes", "") or ""),
            )
        )
    return cases


def dump_case_runs(runs: Iterable[RagCaseRunResult], path: str | Path) -> Path:
    return write_jsonl((run.to_dict() for run in runs), path)


def dump_generated_queries(records: Iterable[GeneratedQueryReviewRecord], path: str | Path) -> Path:
    return write_jsonl((record.to_dict() for record in records), path)
