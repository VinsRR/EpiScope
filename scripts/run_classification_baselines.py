"""Run simple classification baselines over a paper manifest CSV.

The output layout intentionally mirrors run_repeated_experiments.py:

  <base-output-dir>/<task>/<baseline>/<strategy>/<text-scope>/<run-id>/final_1.tsv

LLM baselines add a temperature folder between <baseline> and <strategy>.

That lets eval/scripts/run_classification_eval.py consume baseline outputs
without any special-case code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from classification.baselines import (
    BERTOPIC_SEMISUPERVISED_BASELINES,
    FrozenTransformerEmbedder,
    MajorityLabelBaseline,
    MetadataOnlyLLMBaseline,
    PrototypeSimilarityBaseline,
    SUPERVISED_BASELINES,
    SUPERVISED_CLASSIFIER_BASELINES,
    SUPERVISED_FROZEN_BASELINES,
    SUPERVISED_TOPIC_BASELINES,
    SupervisedCVBaseline,
    SupervisedFrozenEmbeddingBaseline,
    TOPIC_MODEL_BASELINES,
    TopicModelBaseline,
    UNSUPERVISED_TOPIC_BASELINES,
    ZERO_SHOT_TOPIC_BASELINES,
    build_llm_generator,
    classifier_config,
    ground_truth_column,
    is_supervised_baseline,
    is_supervised_frozen_baseline,
    is_supervised_topic_baseline,
    metadata_from_mapping,
    result_to_tsv_row,
    stable_hash,
    task_slug,
)
from classification.baselines.common import slugify
from classification.baselines.llm import (
    usage_delta,
    usage_snapshot_from_baseline,
)
from episcope.schemas import PaperMetadata
from episcope.settings import AppSettings

CLASSIFIER_KINDS = ("paper_type", "data_accessibility", "data_type", "geo")
LIGHT_TOPIC_BASELINES = ("lsa", "plsa", "lda", "nmf")
ZERO_SHOT_BASELINES = ("majority", "prototype_similarity", *ZERO_SHOT_TOPIC_BASELINES)
DEFAULT_BASELINES = ("majority", "prototype_similarity", *LIGHT_TOPIC_BASELINES)
NON_LLM_BASELINES = (*ZERO_SHOT_BASELINES, *TOPIC_MODEL_BASELINES, *SUPERVISED_BASELINES)
BASELINE_KINDS = (*NON_LLM_BASELINES, "metadata_llm")
BASELINE_ALIASES = {
    "prototype": "prototype_similarity",
    "zero_shot_llm": "metadata_llm",
    "bertopic_supervised": "supervised_bertopic",
}
BASELINE_CHOICES = (
    *BASELINE_KINDS,
    *BASELINE_ALIASES,
    "all",
    "all_default",
    "all_non_llm",
    "all_zero_shot",
    "all_unsupervised",
    "all_topics",         # kept as alias for all_unsupervised
    "all_supervised",
    "all_supervised_classifiers",
    "all_supervised_topics",
    "all_supervised_frozen",
)


def normalize_baseline_kind(baseline_kind: str) -> str:
    return BASELINE_ALIASES.get(baseline_kind, baseline_kind)


@dataclass(frozen=True)
class Settings:
    input_csv: str = "sampled_papers_full.csv"
    input_csv_sep: str = "\t"
    ground_truth_csv: str | None = "sampled_papers_full.csv"
    ground_truth_csv_sep: str = "\t"
    base_output_dir: str = "outputs/baselines"
    strategy_name: str = "grobid"
    mongo_uri: str | None = None
    mongo_db_name: str = "episcope_academic_db"
    text_scope: str = "full_text"
    classifier_kinds: tuple[str, ...] = CLASSIFIER_KINDS
    baseline_kinds: tuple[str, ...] = DEFAULT_BASELINES
    repeats: int = 1
    checkpoint_every: int = 25
    overwrite: bool = False
    majority_fit_mode: str = "leave_one_out"
    prototype_multilabel_ratio: float = 0.92
    prototype_min_score: float = 0.03
    prototype_embedding_model: str | None = "allenai-specter"
    topic_n_topics: int = 10
    topic_k_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    topic_k_values: tuple[int, ...] = ()
    topic_max_features: int = 5000
    topic_min_df: float = 1.0
    topic_max_df: float = 0.95
    topic_random_state: int = 13
    bertopic_semisupervised_label_fraction: float = 0.5
    supervised_cv_modes: tuple[str, ...] = ("kfold",)
    supervised_cv_folds: int = 5
    supervised_threshold: float = 0.5
    supervised_frozen_model: str = "allenai/scibert_scivocab_uncased"
    supervised_frozen_text_source: str = "metadata"  # "metadata" or "full_text"
    supervised_frozen_pooling: str = "mean"  # "mean" or "cls"
    supervised_frozen_max_length: int = 512
    supervised_frozen_batch_size: int = 8
    supervised_frozen_normalize: bool = True
    supervised_frozen_cache_dir: str = "outputs/.frozen_embedding_cache"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise IOError(f"Temporary TSV was not written: {tmp}")
    tmp.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def load_records(path: str | Path, sep: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path, sep=sep)
    if "paper_id" not in df.columns:
        raise ValueError(f"{path} must contain a paper_id column.")
    return df.to_dict(orient="records")


def paths_for_repeat(run_dir: Path, repeat_idx: int) -> dict[str, Path]:
    return {
        "checkpoint_tsv": run_dir / f"checkpoint_{repeat_idx}.tsv",
        "final_tsv": run_dir / f"final_{repeat_idx}.tsv",
        "manifest": run_dir / "run_manifest.json",
        "log": run_dir / f"run_{repeat_idx}.log",
    }


def baseline_model_slug(settings: Settings, baseline_kind: str) -> str:
    if baseline_kind == "prototype_similarity" and settings.prototype_embedding_model:
        safe = settings.prototype_embedding_model.replace("/", "-").replace(":", "-")
        return f"prototype_similarity-emb_{safe}"
    if baseline_kind == "metadata_llm":
        return (
            "metadata-llm-"
            + settings.llm_provider
            + "-"
            + settings.llm_model
        )
    if baseline_kind in TOPIC_MODEL_BASELINES:
        if baseline_kind == "bertopic_semisupervised":
            return (
                f"{baseline_kind}-k_{settings.topic_n_topics}"
                f"-labels_{settings.bertopic_semisupervised_label_fraction:g}"
            )
        return f"{baseline_kind}-k_{settings.topic_n_topics}"
    if is_supervised_baseline(baseline_kind):
        cv_slug = (
            f"cv_kfold_{settings.supervised_cv_folds}"
            if settings.supervised_cv_modes == ("kfold",)
            else "cv_leave_one_out"
            if settings.supervised_cv_modes == ("leave_one_out",)
            else "cv_mixed"
        )
        topic_slug = (
            f"-k_{settings.topic_n_topics}"
            if is_supervised_topic_baseline(baseline_kind)
            else ""
        )
        frozen_slug = ""
        if is_supervised_frozen_baseline(baseline_kind):
            model_slug = slugify(settings.supervised_frozen_model)
            frozen_slug = f"-enc_{model_slug}-src_{settings.supervised_frozen_text_source}"
        return f"{baseline_kind}{topic_slug}{frozen_slug}-{cv_slug}"
    return baseline_kind


def build_run_dir(
    settings: Settings,
    *,
    classifier_kind: str,
    baseline_kind: str,
) -> Path:
    baseline_kind = normalize_baseline_kind(baseline_kind)
    identity = {
        "input_csv": settings.input_csv,
        "ground_truth_csv": settings.ground_truth_csv,
        "classifier_kind": classifier_kind,
        "baseline_kind": baseline_kind,
        "strategy_name": settings.strategy_name,
        "text_scope": settings.text_scope,
        "majority_fit_mode": settings.majority_fit_mode,
        "prototype_multilabel_ratio": settings.prototype_multilabel_ratio,
        "prototype_min_score": settings.prototype_min_score,
        "prototype_embedding_model": settings.prototype_embedding_model,
        "topic_n_topics": settings.topic_n_topics,
        "topic_k_multipliers": settings.topic_k_multipliers,
        "topic_k_values": settings.topic_k_values,
        "topic_max_features": settings.topic_max_features,
        "topic_min_df": settings.topic_min_df,
        "topic_max_df": settings.topic_max_df,
        "topic_random_state": settings.topic_random_state,
        "bertopic_semisupervised_label_fraction": settings.bertopic_semisupervised_label_fraction,
        "supervised_cv_modes": settings.supervised_cv_modes,
        "supervised_cv_folds": settings.supervised_cv_folds,
        "supervised_threshold": settings.supervised_threshold,
        "supervised_frozen_model": settings.supervised_frozen_model,
        "supervised_frozen_text_source": settings.supervised_frozen_text_source,
        "supervised_frozen_pooling": settings.supervised_frozen_pooling,
        "supervised_frozen_max_length": settings.supervised_frozen_max_length,
        "supervised_frozen_normalize": settings.supervised_frozen_normalize,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
    }
    run_id = f"{Path(settings.input_csv).stem}-{stable_hash(identity)}"
    run_dir = (
        Path(settings.base_output_dir)
        / task_slug(classifier_kind)
        / baseline_model_slug(settings, baseline_kind).replace("/", "-").replace(":", "-")
    )
    if baseline_kind == "metadata_llm":
        run_dir = run_dir / f"temperature_{settings.llm_temperature}"
    return run_dir / settings.strategy_name / settings.text_scope / run_id


def build_baseline(
    settings: Settings,
    *,
    classifier_kind: str,
    baseline_kind: str,
    records: list[Mapping[str, Any]],
    ground_truth_records: list[Mapping[str, Any]],
):
    baseline_kind = normalize_baseline_kind(baseline_kind)
    topic_min_df = (
        int(settings.topic_min_df)
        if settings.topic_min_df >= 1 and float(settings.topic_min_df).is_integer()
        else settings.topic_min_df
    )
    topic_max_df = (
        int(settings.topic_max_df)
        if settings.topic_max_df > 1 and float(settings.topic_max_df).is_integer()
        else settings.topic_max_df
    )
    if baseline_kind == "majority":
        gt_col = ground_truth_column(classifier_kind)
        return MajorityLabelBaseline.from_records(
            classifier_kind=classifier_kind,
            records=ground_truth_records,
            ground_truth_column=gt_col,
            leave_one_out=settings.majority_fit_mode == "leave_one_out",
        )
    if baseline_kind == "prototype_similarity":
        return PrototypeSimilarityBaseline(
            classifier_kind=classifier_kind,
            records=records,
            multilabel_ratio=settings.prototype_multilabel_ratio,
            min_score=settings.prototype_min_score,
            embedding_model=settings.prototype_embedding_model,
        )
    if baseline_kind in TOPIC_MODEL_BASELINES:
        gt_col = ground_truth_column(classifier_kind)
        return TopicModelBaseline(
            classifier_kind=classifier_kind,
            model_kind=baseline_kind,
            records=records,
            ground_truth_records=ground_truth_records,
            ground_truth_column=gt_col,
            n_topics=settings.topic_n_topics,
            max_features=settings.topic_max_features,
            min_df=topic_min_df,
            max_df=topic_max_df,
            random_state=settings.topic_random_state,
            leave_one_out=settings.majority_fit_mode == "leave_one_out",
            semisupervised_label_fraction=settings.bertopic_semisupervised_label_fraction,
        )
    if is_supervised_frozen_baseline(baseline_kind):
        gt_col = ground_truth_column(classifier_kind)
        embedder = FrozenTransformerEmbedder(
            model_name=settings.supervised_frozen_model,
            pooling=settings.supervised_frozen_pooling,
            max_length=settings.supervised_frozen_max_length,
            batch_size=settings.supervised_frozen_batch_size,
            cache_dir=settings.supervised_frozen_cache_dir or None,
        )
        return SupervisedFrozenEmbeddingBaseline(
            embedder=embedder,
            text_source=settings.supervised_frozen_text_source,
            normalize_features=settings.supervised_frozen_normalize,
            classifier_kind=classifier_kind,
            baseline_kind=baseline_kind,
            records=records,
            ground_truth_records=ground_truth_records,
            ground_truth_column=gt_col,
            cv_mode=settings.supervised_cv_modes[0],
            cv_folds=settings.supervised_cv_folds,
            random_state=settings.topic_random_state,
            threshold=settings.supervised_threshold,
        )
    if is_supervised_baseline(baseline_kind):
        gt_col = ground_truth_column(classifier_kind)
        return SupervisedCVBaseline(
            classifier_kind=classifier_kind,
            baseline_kind=baseline_kind,
            records=records,
            ground_truth_records=ground_truth_records,
            ground_truth_column=gt_col,
            cv_mode=settings.supervised_cv_modes[0],
            cv_folds=settings.supervised_cv_folds,
            n_topics=settings.topic_n_topics,
            max_features=settings.topic_max_features,
            min_df=topic_min_df,
            max_df=topic_max_df,
            random_state=settings.topic_random_state,
            threshold=settings.supervised_threshold,
        )
    if baseline_kind == "metadata_llm":
        generator = build_llm_generator(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        return MetadataOnlyLLMBaseline(
            classifier_kind=classifier_kind,
            generator=generator,
        )
    raise ValueError(f"Unknown baseline_kind={baseline_kind!r}.")


def task_label_count(classifier_kind: str) -> int:
    labels = {
        label
        for label in classifier_config(classifier_kind).classification_mapping.values()
        if getattr(label, "name", str(label)).upper() != "UNCLEAR"
    }
    return max(1, len(labels))


def topic_k_values_for_task(settings: Settings, classifier_kind: str) -> tuple[int, ...]:
    if settings.topic_k_values:
        values = settings.topic_k_values
    elif settings.topic_k_multipliers:
        label_count = task_label_count(classifier_kind)
        values = tuple(
            max(2, int(round(label_count * multiplier)))
            for multiplier in settings.topic_k_multipliers
        )
    else:
        values = (settings.topic_n_topics,)

    deduped: list[int] = []
    for value in values:
        clean = max(2, int(value))
        if clean not in deduped:
            deduped.append(clean)
    return tuple(deduped)


def settings_for_baseline(
    settings: Settings,
    *,
    classifier_kind: str,
    baseline_kind: str,
) -> tuple[Settings, ...]:
    baseline_kind = normalize_baseline_kind(baseline_kind)
    cv_modes = settings.supervised_cv_modes or ("kfold",)
    if is_supervised_baseline(baseline_kind):
        return tuple(
            dataclasses.replace(
                settings,
                supervised_cv_modes=(cv_mode,),
            )
            for cv_mode in cv_modes
        )
    if baseline_kind in TOPIC_MODEL_BASELINES:
        return tuple(
            dataclasses.replace(settings, topic_n_topics=k)
            for k in topic_k_values_for_task(settings, classifier_kind)
        )
    return (settings,)


def _section_text(section: Any) -> str:
    if isinstance(section, Mapping):
        return str(section.get("content", "") or "").strip()
    return str(getattr(section, "content", "") or "").strip()


def _section_title(section: Any) -> str:
    if isinstance(section, Mapping):
        return str(section.get("title", "") or "").strip()
    return str(getattr(section, "title", "") or "").strip()


def _section_type(section: Any) -> str:
    if isinstance(section, Mapping):
        return str(section.get("section_type", "") or "").strip()
    return str(getattr(section, "section_type", "") or "").strip()


def _abstract_sections(sections: Iterable[Any]) -> list[Any]:
    abstract_sections = []
    for section in sections:
        title = _section_title(section).lower()
        section_type = _section_type(section).lower()
        if "abstract" in title or "abstract" in section_type:
            abstract_sections.append(section)
    return abstract_sections


def paper_text_from_db(
    db: Any,
    paper_id: str,
    strategy_name: str,
    *,
    text_scope: str = "full_text",
) -> tuple[PaperMetadata, str, dict[str, Any]]:
    metadata = db.get_paper_metadata(paper_id, strategy_name)
    if metadata is None:
        metadata = PaperMetadata(doi=paper_id)

    sections = db.retrieve(paper_id, "sections", strategy_name) or []
    abstract_sections = _abstract_sections(sections)
    blocks: list[str] = []
    title = str(getattr(metadata, "title", "") or "").strip()
    abstract = str(getattr(metadata, "abstract", "") or "").strip()
    if text_scope == "abstract":
        abstract_blocks = [abstract]
        abstract_blocks.extend(_section_text(section) for section in abstract_sections)
        text = "\n\n".join(
            block for block in abstract_blocks if block and block.strip()
        ).strip()
        stats = {
            "text_source": "mongo_abstract",
            "text_scope": text_scope,
            "section_count": len(sections),
            "used_section_count": len(
                [section for section in abstract_sections if _section_text(section)]
            ),
            "metadata_abstract_char_count": len(abstract),
            "abstract_section_count": len(abstract_sections),
            "char_count": len(text),
        }
        return metadata, text, stats
    if text_scope != "full_text":
        raise ValueError("text_scope must be one of: full_text, abstract")

    if title:
        blocks.append(title)
    if abstract:
        blocks.append(abstract)
    for section in sections:
        heading = _section_title(section)
        content = _section_text(section)
        if not content:
            continue
        blocks.append(f"{heading}\n{content}" if heading else content)

    text = "\n\n".join(block for block in blocks if block.strip()).strip()
    stats = {
        "text_source": "mongo_full_text",
        "text_scope": text_scope,
        "section_count": len(sections),
        "used_section_count": sum(1 for section in sections if _section_text(section)),
        "metadata_abstract_char_count": len(abstract),
        "abstract_section_count": len(abstract_sections),
        "char_count": len(text),
    }
    return metadata, text, stats


def build_mongo_db(settings: Settings):
    from episcope.db.mongo_academic_db import MongoAcademicDB

    if not settings.mongo_uri:
        raise RuntimeError(
            "Mongo URI is not configured. Set MONGO_URI in .env or the environment."
        )
    return MongoAcademicDB(
        uri=settings.mongo_uri,
        db_name=settings.mongo_db_name,
    )


def records_with_paper_text(settings: Settings, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    db = build_mongo_db(settings)
    enriched = []
    for record in records:
        paper_id = str(record["paper_id"])
        metadata, text, stats = paper_text_from_db(
            db,
            paper_id,
            settings.strategy_name,
            text_scope=settings.text_scope,
        )
        skip_reason = None
        if not text:
            skip_reason = f"missing_{settings.text_scope}"
        enriched.append(
            {
                **record,
                "_metadata": metadata,
                "_metadata_text": text,
                "_text_stats": stats,
                "_text_scope": settings.text_scope,
                "_skip_reason": skip_reason,
            }
        )
    return enriched


def skipped_result_row(
    paper_id: str,
    *,
    classifier_kind: str,
    baseline_kind: str,
    reason: str,
    text_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "paper_id": paper_id,
        "classifier_kind": classifier_kind,
        "baseline_name": baseline_kind,
        "classification": "__SKIPPED__",
        "reasoning": reason,
        "confidence": 0.0,
        "raw_response": "",
        "messages": "",
        "evaluation_status": "skipped",
        "skip_reason": reason,
    }
    text_stats = text_stats or {}
    row.update(
        {
            "text_source": text_stats.get("text_source"),
            "text_scope": text_stats.get("text_scope"),
            "paper_text_char_count": text_stats.get("char_count"),
            "paper_section_count": text_stats.get("section_count"),
            "paper_used_section_count": text_stats.get("used_section_count"),
            "metadata_abstract_char_count": text_stats.get("metadata_abstract_char_count"),
            "abstract_section_count": text_stats.get("abstract_section_count"),
        }
    )
    return row


def metadata_for_record(record: Mapping[str, Any]) -> PaperMetadata:
    metadata = record.get("_metadata")
    text = str(record.get("_metadata_text") or "")
    text_scope = str(record.get("_text_scope") or "")
    if isinstance(metadata, PaperMetadata):
        if text_scope == "abstract":
            return PaperMetadata(
                title="",
                abstract=text or metadata.abstract,
                authors=[],
                publication_year=None,
                journal="",
                doi=metadata.doi,
                keywords=[],
                first_author=None,
                file_path=None,
            )
        return PaperMetadata(
            title=metadata.title,
            abstract=text or metadata.abstract,
            authors=list(metadata.authors or []),
            publication_year=metadata.publication_year,
            journal=metadata.journal,
            doi=metadata.doi,
            keywords=list(metadata.keywords or []),
            first_author=metadata.first_author,
            file_path=metadata.file_path,
        )
    base = metadata_from_mapping(record)
    if text:
        base.abstract = text
    return base


def run_once(
    settings: Settings,
    *,
    classifier_kind: str,
    baseline_kind: str,
    repeat_idx: int,
    records: list[dict[str, Any]],
    ground_truth_records: list[dict[str, Any]],
) -> Path:
    baseline_kind = normalize_baseline_kind(baseline_kind)
    run_dir = build_run_dir(
        settings, classifier_kind=classifier_kind, baseline_kind=baseline_kind
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for_repeat(run_dir, repeat_idx)
    if paths["final_tsv"].exists() and not settings.overwrite:
        print(f"[OK] final already exists: {paths['final_tsv']}")
        return paths["final_tsv"]

    enriched_records = records_with_paper_text(settings, records)
    fit_records = [
        record for record in enriched_records if not record.get("_skip_reason")
    ]
    skipped_records = [
        record for record in enriched_records if record.get("_skip_reason")
    ]
    if skipped_records:
        sample = ", ".join(str(record["paper_id"]) for record in skipped_records[:5])
        message = (
            f"skipping {len(skipped_records)} paper(s) with no usable "
            f"{settings.text_scope} text. Examples: {sample}"
        )
        append_log(paths["log"], message)
        print(f"[WARN] {message}")

    baseline = None
    if fit_records:
        baseline = build_baseline(
            settings,
            classifier_kind=classifier_kind,
            baseline_kind=baseline_kind,
            records=fit_records,
            ground_truth_records=ground_truth_records,
        )

    checkpoint = (
        pd.read_csv(paths["checkpoint_tsv"], sep="\t")
        if paths["checkpoint_tsv"].exists() and not settings.overwrite
        else pd.DataFrame()
    )
    done_ids = (
        set(checkpoint["paper_id"].astype(str))
        if not checkpoint.empty and "paper_id" in checkpoint
        else set()
    )
    rows: list[dict[str, Any]] = []
    total = len(enriched_records)
    append_log(paths["log"], f"start baseline={baseline_kind} task={classifier_kind}")

    for idx, record in enumerate(enriched_records, start=1):
        paper_id = str(record["paper_id"])
        if paper_id in done_ids:
            print(f"[{idx}/{total}] skip {paper_id} (checkpoint)")
            continue
        skip_reason = record.get("_skip_reason")
        if skip_reason and not getattr(baseline, "skip_if_no_text", True):
            # Baseline declared it can work without body text — only skip if
            # there is genuinely no metadata object to work from.
            if record.get("_metadata") is not None:
                skip_reason = None
        if skip_reason:
            print(f"[{idx}/{total}] skip {paper_id} ({skip_reason})")
            rows.append(
                skipped_result_row(
                    paper_id,
                    classifier_kind=classifier_kind,
                    baseline_kind=baseline_kind,
                    reason=str(skip_reason),
                    text_stats=record.get("_text_stats")
                    if isinstance(record.get("_text_stats"), Mapping)
                    else None,
                )
            )
            if len(rows) >= settings.checkpoint_every:
                merged = pd.concat([checkpoint, pd.DataFrame(rows)], ignore_index=True)
                merged = merged.drop_duplicates(subset=["paper_id"], keep="first")
                atomic_write_tsv(merged, paths["checkpoint_tsv"])
                checkpoint = merged
                done_ids = set(checkpoint["paper_id"].astype(str))
                append_log(paths["log"], f"checkpoint rows={len(checkpoint)}/{total}")
                rows.clear()
            continue
        if baseline is None:
            raise RuntimeError("No non-skipped papers are available for this run.")
        print(f"[{idx}/{total}] {baseline_kind} {classifier_kind} {paper_id}")
        metadata = metadata_for_record(record)
        usage_before = usage_snapshot_from_baseline(baseline)
        prediction = baseline.predict(paper_id, metadata, record)
        usage_after = usage_snapshot_from_baseline(baseline)
        row = result_to_tsv_row(
            paper_id,
            prediction,
            classifier_kind=classifier_kind,
            baseline_name=baseline_kind,
        )
        row["evaluation_status"] = "evaluated"
        row["skip_reason"] = ""
        text_stats = record.get("_text_stats") or {}
        if isinstance(text_stats, Mapping):
            row.update(
                {
                    "text_source": text_stats.get("text_source"),
                    "text_scope": text_stats.get("text_scope"),
                    "paper_text_char_count": text_stats.get("char_count"),
                    "paper_section_count": text_stats.get("section_count"),
                    "paper_used_section_count": text_stats.get("used_section_count"),
                    "metadata_abstract_char_count": text_stats.get("metadata_abstract_char_count"),
                    "abstract_section_count": text_stats.get("abstract_section_count"),
                }
            )
        if usage_before is not None and usage_after is not None:
            usage = usage_delta(usage_after, usage_before)
            row.update(
                {
                    "llm_prompt_tokens": usage["prompt_tokens"],
                    "llm_completion_tokens": usage["completion_tokens"],
                    "llm_total_tokens": usage["total_tokens"],
                    "llm_cached_tokens": usage["cached_tokens"],
                    "llm_reasoning_tokens": usage["reasoning_tokens"],
                    "llm_call_count": usage["call_count"],
                }
            )
        rows.append(row)

        if len(rows) >= settings.checkpoint_every:
            merged = pd.concat([checkpoint, pd.DataFrame(rows)], ignore_index=True)
            merged = merged.drop_duplicates(subset=["paper_id"], keep="first")
            atomic_write_tsv(merged, paths["checkpoint_tsv"])
            checkpoint = merged
            done_ids = set(checkpoint["paper_id"].astype(str))
            append_log(paths["log"], f"checkpoint rows={len(checkpoint)}/{total}")
            rows.clear()

    if rows:
        checkpoint = pd.concat([checkpoint, pd.DataFrame(rows)], ignore_index=True)
        checkpoint = checkpoint.drop_duplicates(subset=["paper_id"], keep="first")
        atomic_write_tsv(checkpoint, paths["checkpoint_tsv"])

    if len(checkpoint) != total:
        append_log(paths["log"], f"incomplete rows={len(checkpoint)}/{total}")
        print(f"[WARN] incomplete run: {len(checkpoint)}/{total}")
        return paths["checkpoint_tsv"]

    atomic_write_tsv(checkpoint, paths["final_tsv"])
    n_skipped = int(
        checkpoint.get("evaluation_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("skipped")
        .sum()
    )
    manifest = {
        "classifier_kind": classifier_kind,
        "baseline_kind": baseline_kind,
        "repeat_idx": repeat_idx,
        "n_rows": len(checkpoint),
        "n_predictions": len(checkpoint) - n_skipped,
        "n_skipped": n_skipped,
        "input_csv": settings.input_csv,
        "ground_truth_csv": settings.ground_truth_csv,
        "settings": dataclasses.asdict(settings),
    }
    atomic_write_text(paths["manifest"], json.dumps(manifest, indent=2, sort_keys=True))
    append_log(paths["log"], f"success final={paths['final_tsv']} skipped={n_skipped}")
    print(f"[OK] wrote {paths['final_tsv']} ({len(checkpoint)} rows, {n_skipped} skipped)")
    return paths["final_tsv"]


def expand_arg_values(values: Iterable[str] | None, *, all_values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return all_values
    out: list[str] = []
    for value in values:
        if value in {"all", "all_non_llm"}:
            out.extend(all_values)
        elif value == "all_default":
            out.extend(DEFAULT_BASELINES)
        elif value == "all_zero_shot":
            out.extend(ZERO_SHOT_BASELINES)
        elif value in {"all_unsupervised", "all_topics"}:
            out.extend(UNSUPERVISED_TOPIC_BASELINES)
        elif value == "all_supervised":
            out.extend(SUPERVISED_BASELINES)
        elif value == "all_supervised_classifiers":
            out.extend(SUPERVISED_CLASSIFIER_BASELINES)
        elif value == "all_supervised_topics":
            out.extend(SUPERVISED_TOPIC_BASELINES)
        elif value == "all_supervised_frozen":
            out.extend(SUPERVISED_FROZEN_BASELINES)
        else:
            out.append(normalize_baseline_kind(value))
    deduped = []
    for value in out:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification baselines.")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-csv-sep", default=None)
    parser.add_argument("--ground-truth-csv", default=None)
    parser.add_argument("--ground-truth-csv-sep", default=None)
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--strategy-name", default=None)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--mongo-db-name", default=None)
    parser.add_argument(
        "--text-scope",
        choices=["full_text", "abstract"],
        default=None,
        help=(
            "Which Mongo text to give to the baselines. full_text uses title, "
            "abstract, and sections; abstract uses metadata.abstract plus any "
            "Mongo sections explicitly titled/typed as abstract."
        ),
    )
    parser.add_argument(
        "--classifier-kind",
        action="append",
        choices=[*CLASSIFIER_KINDS, "all"],
        default=None,
    )
    parser.add_argument(
        "--baseline-kind",
        action="append",
        choices=BASELINE_CHOICES,
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--majority-fit-mode",
        choices=["leave_one_out", "all"],
        default=None,
    )
    parser.add_argument("--prototype-multilabel-ratio", type=float, default=None)
    parser.add_argument("--prototype-min-score", type=float, default=None)
    parser.add_argument(
        "--prototype-embedding-model",
        default=None,
        help=(
            "sentence-transformers model name for the prototype baseline. "
            "When set, uses dense embeddings instead of TF-IDF (e.g. "
            "'all-MiniLM-L6-v2' or 'allenai-specter')."
        ),
    )
    parser.add_argument("--topic-n-topics", type=int, default=None)
    parser.add_argument(
        "--topic-k-multiplier",
        action="append",
        type=float,
        default=None,
        help=(
            "Run unsupervised topic baselines with K = round(L * multiplier), "
            "where L is the number of task labels. Repeat this flag for a K "
            "sweep. Defaults to 0.5, 1, 2, and 4 times L."
        ),
    )
    parser.add_argument(
        "--topic-k-value",
        action="append",
        type=int,
        default=None,
        help=(
            "Explicit topic count K for unsupervised topic baselines. Repeat "
            "for a sweep. Overrides --topic-k-multiplier when provided."
        ),
    )
    parser.add_argument("--topic-max-features", type=int, default=None)
    parser.add_argument("--topic-min-df", type=float, default=None)
    parser.add_argument("--topic-max-df", type=float, default=None)
    parser.add_argument("--topic-random-state", type=int, default=None)
    parser.add_argument(
        "--bertopic-semisupervised-label-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of gold labels exposed to BERTopic semi-supervised mode; "
            "remaining papers are passed as unlabeled (-1)."
        ),
    )
    parser.add_argument(
        "--supervised-cv-mode",
        action="append",
        choices=["kfold", "leave_one_out"],
        default=None,
        help=(
            "Cross-validation mode for supervised baselines. Defaults to "
            "kfold; use leave_one_out for leave-one-out CV. Repeat to run both."
        ),
    )
    parser.add_argument("--supervised-cv-folds", type=int, default=None)
    parser.add_argument("--supervised-threshold", type=float, default=None)
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "openrouter", "openai", "ollama"],
        default=None,
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    return parser.parse_args()


def merge_settings(args: argparse.Namespace) -> Settings:
    app_settings = AppSettings.from_env()
    base = Settings(
        strategy_name=app_settings.strategy_name,
        mongo_uri=app_settings.mongo_uri,
        mongo_db_name=app_settings.mongo_db_name,
    )
    data = dataclasses.asdict(base)
    for field in (
        "input_csv",
        "input_csv_sep",
        "ground_truth_csv",
        "ground_truth_csv_sep",
        "base_output_dir",
        "strategy_name",
        "mongo_uri",
        "mongo_db_name",
        "text_scope",
        "repeats",
        "checkpoint_every",
        "majority_fit_mode",
        "prototype_multilabel_ratio",
        "prototype_min_score",
        "prototype_embedding_model",
        "topic_n_topics",
        "topic_max_features",
        "topic_min_df",
        "topic_max_df",
        "topic_random_state",
        "bertopic_semisupervised_label_fraction",
        "supervised_cv_folds",
        "supervised_threshold",
        "llm_provider",
        "llm_model",
        "llm_temperature",
    ):
        value = getattr(args, field)
        if value is not None:
            data[field] = value
    if args.topic_k_multiplier is not None:
        data["topic_k_multipliers"] = tuple(args.topic_k_multiplier)
    if args.topic_k_value is not None:
        data["topic_k_values"] = tuple(args.topic_k_value)
    if args.topic_n_topics is not None and args.topic_k_multiplier is None and args.topic_k_value is None:
        data["topic_k_multipliers"] = ()
        data["topic_k_values"] = ()
    if args.supervised_cv_mode is not None:
        data["supervised_cv_modes"] = tuple(args.supervised_cv_mode)
    if args.overwrite:
        data["overwrite"] = True
    data["classifier_kinds"] = expand_arg_values(
        args.classifier_kind,
        all_values=CLASSIFIER_KINDS,
    )
    if args.baseline_kind is not None:
        data["baseline_kinds"] = expand_arg_values(
            args.baseline_kind,
            all_values=NON_LLM_BASELINES,
        )
    return Settings(**data)


def main() -> None:
    settings = merge_settings(parse_args())
    records = load_records(settings.input_csv, settings.input_csv_sep)
    if settings.ground_truth_csv:
        ground_truth_records = load_records(
            settings.ground_truth_csv,
            settings.ground_truth_csv_sep,
        )
    else:
        ground_truth_records = records

    failures = 0
    for classifier_kind in settings.classifier_kinds:
        for baseline_kind in settings.baseline_kinds:
            for run_settings in settings_for_baseline(
                settings,
                classifier_kind=classifier_kind,
                baseline_kind=baseline_kind,
            ):
                for repeat_idx in range(1, run_settings.repeats + 1):
                    try:
                        k_suffix = (
                            f" k={run_settings.topic_n_topics}"
                            if normalize_baseline_kind(baseline_kind) in TOPIC_MODEL_BASELINES
                            or is_supervised_topic_baseline(normalize_baseline_kind(baseline_kind))
                            else ""
                        )
                        cv_suffix = (
                            f" cv={run_settings.supervised_cv_modes[0]}"
                            if is_supervised_baseline(normalize_baseline_kind(baseline_kind))
                            else ""
                        )
                        print(
                            f"\n=== task={classifier_kind} baseline={baseline_kind}"
                            f"{k_suffix}{cv_suffix} repeat={repeat_idx}/{run_settings.repeats} ==="
                        )
                        run_once(
                            run_settings,
                            classifier_kind=classifier_kind,
                            baseline_kind=baseline_kind,
                            repeat_idx=repeat_idx,
                            records=records,
                            ground_truth_records=ground_truth_records,
                        )
                    except Exception as exc:
                        failures += 1
                        print(
                            f"[ERROR] task={classifier_kind} baseline={baseline_kind}: {exc!r}",
                            file=sys.stderr,
                        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
