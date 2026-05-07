"""Run simple classification baselines over a paper manifest CSV.

The output layout intentionally mirrors run_repeated_experiments.py:

  <base-output-dir>/<task>/<baseline>/<strategy>/<run-id>/final_1.tsv

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

from episcope.workflows.classification.baselines import (
    BASE_TOPIC_MODEL_KINDS,
    GuidedTopicModelBaseline,
    MajorityLabelBaseline,
    MetadataOnlyLLMBaseline,
    PrototypeSimilarityBaseline,
    TOPIC_MAJORITY_BASELINES,
    TOPIC_MODEL_BASELINES,
    TopicModelBaseline,
    base_topic_model_kind,
    build_llm_generator,
    ground_truth_column,
    metadata_from_mapping,
    result_to_tsv_row,
    stable_hash,
    task_slug,
)
from episcope.workflows.classification.baselines.llm import (
    usage_delta,
    usage_snapshot_from_baseline,
)
from episcope.schemas import PaperMetadata
from episcope.settings import AppSettings

CLASSIFIER_KINDS = ("paper_type", "data_accessibility", "data_type", "geo")
LIGHT_TOPIC_BASELINES = ("guided_lsa", "guided_plsa", "guided_lda", "guided_nmf")
DEFAULT_BASELINES = ("majority", "prototype_similarity", *LIGHT_TOPIC_BASELINES)
NON_LLM_BASELINES = ("majority", "prototype_similarity", *TOPIC_MODEL_BASELINES)
BASELINE_KINDS = (*NON_LLM_BASELINES, "metadata_llm")
BASELINE_ALIASES = {
    "prototype": "prototype_similarity",
    "zero_shot_llm": "metadata_llm",
    **{kind: f"guided_{kind}" for kind in BASE_TOPIC_MODEL_KINDS},
}
BASELINE_CHOICES = (
    *BASELINE_KINDS,
    *TOPIC_MAJORITY_BASELINES,
    *BASELINE_ALIASES,
    "all",
    "all_default",
    "all_non_llm",
    "all_topics",
    "all_topic_majority",
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
    classifier_kinds: tuple[str, ...] = CLASSIFIER_KINDS
    baseline_kinds: tuple[str, ...] = DEFAULT_BASELINES
    repeats: int = 1
    checkpoint_every: int = 25
    overwrite: bool = False
    majority_fit_mode: str = "leave_one_out"
    prototype_multilabel_ratio: float = 0.92
    prototype_min_score: float = 0.03
    topic_n_topics: int = 10
    topic_max_features: int = 5000
    topic_min_df: float = 1.0
    topic_max_df: float = 0.95
    topic_random_state: int = 13
    guided_multilabel_ratio: float = 0.6
    guided_min_score: float = 0.05
    guided_max_labels: int = 4
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
    if baseline_kind == "metadata_llm":
        return (
            "metadata-llm-"
            + settings.llm_provider
            + "-"
            + settings.llm_model
        )
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
        "majority_fit_mode": settings.majority_fit_mode,
        "prototype_multilabel_ratio": settings.prototype_multilabel_ratio,
        "prototype_min_score": settings.prototype_min_score,
        "topic_n_topics": settings.topic_n_topics,
        "topic_max_features": settings.topic_max_features,
        "topic_min_df": settings.topic_min_df,
        "topic_max_df": settings.topic_max_df,
        "topic_random_state": settings.topic_random_state,
        "guided_multilabel_ratio": settings.guided_multilabel_ratio,
        "guided_min_score": settings.guided_min_score,
        "guided_max_labels": settings.guided_max_labels,
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
    return run_dir / settings.strategy_name / run_id


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
        )
    if baseline_kind in TOPIC_MODEL_BASELINES:
        return GuidedTopicModelBaseline(
            classifier_kind=classifier_kind,
            model_kind=base_topic_model_kind(baseline_kind),
            records=records,
            n_topics=settings.topic_n_topics,
            max_features=settings.topic_max_features,
            min_df=topic_min_df,
            max_df=topic_max_df,
            random_state=settings.topic_random_state,
            multilabel_ratio=settings.guided_multilabel_ratio,
            min_score=settings.guided_min_score,
            max_labels=settings.guided_max_labels,
        )
    if baseline_kind in TOPIC_MAJORITY_BASELINES:
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


def _section_text(section: Any) -> str:
    if isinstance(section, Mapping):
        return str(section.get("content", "") or "").strip()
    return str(getattr(section, "content", "") or "").strip()


def _section_title(section: Any) -> str:
    if isinstance(section, Mapping):
        return str(section.get("title", "") or "").strip()
    return str(getattr(section, "title", "") or "").strip()


def paper_text_from_db(db: Any, paper_id: str, strategy_name: str) -> tuple[PaperMetadata, str, dict[str, Any]]:
    metadata = db.get_paper_metadata(paper_id, strategy_name)
    if metadata is None:
        metadata = PaperMetadata(doi=paper_id)

    sections = db.retrieve(paper_id, "sections", strategy_name) or []

    blocks: list[str] = []
    title = str(getattr(metadata, "title", "") or "").strip()
    abstract = str(getattr(metadata, "abstract", "") or "").strip()
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
        "text_source": "mongo",
        "section_count": len(sections),
        "used_section_count": sum(1 for section in sections if _section_text(section)),
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
    missing: list[str] = []
    for record in records:
        paper_id = str(record["paper_id"])
        metadata, text, stats = paper_text_from_db(db, paper_id, settings.strategy_name)
        if not text:
            missing.append(paper_id)
        enriched.append(
            {
                **record,
                "_metadata": metadata,
                "_metadata_text": text,
                "_text_stats": stats,
            }
        )
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"No paper text found in Mongo for {len(missing)} document(s) "
            f"using strategy={settings.strategy_name!r}. Examples: {sample}"
        )
    return enriched


def metadata_for_record(record: Mapping[str, Any]) -> PaperMetadata:
    metadata = record.get("_metadata")
    text = str(record.get("_metadata_text") or "")
    if isinstance(metadata, PaperMetadata):
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

    fit_records = records_with_paper_text(settings, records)

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
    total = len(fit_records)
    append_log(paths["log"], f"start baseline={baseline_kind} task={classifier_kind}")

    for idx, record in enumerate(fit_records, start=1):
        paper_id = str(record["paper_id"])
        if paper_id in done_ids:
            print(f"[{idx}/{total}] skip {paper_id} (checkpoint)")
            continue
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
        text_stats = record.get("_text_stats") or {}
        if isinstance(text_stats, Mapping):
            row.update(
                {
                    "text_source": text_stats.get("text_source"),
                    "paper_text_char_count": text_stats.get("char_count"),
                    "paper_section_count": text_stats.get("section_count"),
                    "paper_used_section_count": text_stats.get("used_section_count"),
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
    manifest = {
        "classifier_kind": classifier_kind,
        "baseline_kind": baseline_kind,
        "repeat_idx": repeat_idx,
        "n_predictions": len(checkpoint),
        "input_csv": settings.input_csv,
        "ground_truth_csv": settings.ground_truth_csv,
        "settings": dataclasses.asdict(settings),
    }
    atomic_write_text(paths["manifest"], json.dumps(manifest, indent=2, sort_keys=True))
    append_log(paths["log"], f"success final={paths['final_tsv']}")
    print(f"[OK] wrote {paths['final_tsv']} ({len(checkpoint)} rows)")
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
        elif value == "all_topics":
            out.extend(TOPIC_MODEL_BASELINES)
        elif value == "all_topic_majority":
            out.extend(TOPIC_MAJORITY_BASELINES)
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
    parser.add_argument("--topic-n-topics", type=int, default=None)
    parser.add_argument("--topic-max-features", type=int, default=None)
    parser.add_argument("--topic-min-df", type=float, default=None)
    parser.add_argument("--topic-max-df", type=float, default=None)
    parser.add_argument("--topic-random-state", type=int, default=None)
    parser.add_argument("--guided-multilabel-ratio", type=float, default=None)
    parser.add_argument("--guided-min-score", type=float, default=None)
    parser.add_argument("--guided-max-labels", type=int, default=None)
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
        "repeats",
        "checkpoint_every",
        "majority_fit_mode",
        "prototype_multilabel_ratio",
        "prototype_min_score",
        "topic_n_topics",
        "topic_max_features",
        "topic_min_df",
        "topic_max_df",
        "topic_random_state",
        "guided_multilabel_ratio",
        "guided_min_score",
        "guided_max_labels",
        "llm_provider",
        "llm_model",
        "llm_temperature",
    ):
        value = getattr(args, field)
        if value is not None:
            data[field] = value
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
            for repeat_idx in range(1, settings.repeats + 1):
                try:
                    print(
                        f"\n=== task={classifier_kind} baseline={baseline_kind} "
                        f"repeat={repeat_idx}/{settings.repeats} ==="
                    )
                    run_once(
                        settings,
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
