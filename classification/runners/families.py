from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from typing import Any

from episcope.settings import AppSettings

import scripts.run_classification_baselines as base


FAMILY_BASELINES = {
    "zero_shot": ("majority", "prototype_similarity", "nli_zero_shot", *base.ZERO_SHOT_TOPIC_BASELINES),
    "unsupervised": base.UNSUPERVISED_TOPIC_BASELINES,
    "supervised": (*base.SUPERVISED_BASELINES, *base.BERTOPIC_SEMISUPERVISED_BASELINES),
    "frozen": base.SUPERVISED_FROZEN_BASELINES,
    "llm": ("metadata_llm", "random_chunk_llm"),
    # Legacy aliases kept so old invocations don't break.
    "simple": ("majority", "prototype_similarity"),
    "topic": base.TOPIC_MODEL_BASELINES,
}


def _add_frozen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--supervised-frozen-model",
        default=None,
        help="HF model id for the frozen encoder (e.g. allenai/scibert_scivocab_uncased).",
    )
    parser.add_argument(
        "--supervised-frozen-text-source",
        choices=["metadata", "full_text"],
        default=None,
        help="Encode 'title [SEP] abstract' (metadata) or the full paper text in chunks (full_text).",
    )
    parser.add_argument(
        "--supervised-frozen-pooling",
        choices=["mean", "cls"],
        default=None,
    )
    parser.add_argument("--supervised-frozen-max-length", type=int, default=None)
    parser.add_argument("--supervised-frozen-batch-size", type=int, default=None)
    parser.add_argument(
        "--supervised-frozen-no-normalize",
        action="store_true",
        help="Disable per-fold StandardScaler on the embeddings.",
    )
    parser.add_argument("--supervised-frozen-cache-dir", default=None)


def family_parser(*, family: str, description: str) -> argparse.ArgumentParser:
    if family not in FAMILY_BASELINES:
        raise ValueError(f"Unknown baseline family {family!r}.")
    aliases = tuple(
        alias
        for alias, target in base.BASELINE_ALIASES.items()
        if target in FAMILY_BASELINES[family]
    )
    choices = (*FAMILY_BASELINES[family], *aliases)
    parser = argparse.ArgumentParser(description=description)
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
    )
    parser.add_argument(
        "--classifier-kind",
        action="append",
        choices=[*base.CLASSIFIER_KINDS, "all"],
        default=None,
    )
    parser.add_argument(
        "--baseline-kind",
        action="append",
        choices=[*choices, "all"],
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    if family in ("zero_shot", "simple"):
        parser.add_argument("--majority-fit-mode", choices=["leave_one_out", "all"], default=None)
        parser.add_argument("--prototype-multilabel-ratio", type=float, default=None)
        parser.add_argument("--prototype-min-score", type=float, default=None)
        parser.add_argument(
            "--prototype-embedding-model",
            default=None,
            help=(
                "sentence-transformers model name for the prototype baseline. "
                "When set, uses dense embeddings instead of TF-IDF."
            ),
        )
        parser.add_argument(
            "--nli-model",
            default=None,
            help="HF NLI checkpoint for the nli_zero_shot baseline.",
        )
        parser.add_argument(
            "--nli-hypothesis-source",
            choices=["label_description", "label_name"],
            default=None,
            help=(
                "What text to use as the NLI hypothesis. 'label_description' "
                "(default) uses the template paragraphs from the classifier "
                "config (same label-information access as EpiScope's prompt); "
                "'label_name' uses the canonical Yin-et-al template."
            ),
        )
        parser.add_argument(
            "--nli-hypothesis-template",
            default=None,
            help="Template used when --nli-hypothesis-source=label_name. Must contain {label}.",
        )
        parser.add_argument(
            "--nli-threshold",
            type=float,
            default=None,
            help="P(entailment) threshold for multi-label NLI selection (default 0.5).",
        )
        parser.add_argument("--nli-max-length", type=int, default=None)
        parser.add_argument("--nli-batch-size", type=int, default=None)
        parser.add_argument("--nli-cache-dir", default=None)
        add_topic_args(parser, include_k_sweep=True)
    if family in ("unsupervised", "topic"):
        add_topic_args(parser, include_k_sweep=True)
    if family == "supervised":
        add_topic_args(parser, include_k_sweep=False)
        parser.add_argument(
            "--supervised-cv-mode",
            action="append",
            choices=["kfold", "leave_one_out"],
            default=None,
        )
        parser.add_argument("--supervised-cv-folds", type=int, default=None)
        parser.add_argument("--supervised-threshold", type=float, default=None)
        # Frozen transformer embedding baselines
        _add_frozen_args(parser)
    if family == "frozen":
        parser.add_argument(
            "--supervised-cv-mode",
            action="append",
            choices=["kfold", "leave_one_out"],
            default=None,
        )
        parser.add_argument("--supervised-cv-folds", type=int, default=None)
        parser.add_argument("--supervised-threshold", type=float, default=None)
        _add_frozen_args(parser)
    if family == "llm":
        parser.add_argument(
            "--llm-provider",
            choices=["gemini", "openrouter", "openai", "ollama"],
            default=None,
        )
        parser.add_argument("--llm-model", default=None)
        parser.add_argument("--llm-temperature", type=float, default=None)
        parser.add_argument(
            "--random-chunk-k",
            type=int,
            default=None,
            help="Number of body chunks to sample for --baseline-kind random_chunk_llm.",
        )
        parser.add_argument(
            "--random-chunk-seed",
            type=int,
            default=None,
            help="Base seed for per-paper random chunk sampling.",
        )
        parser.add_argument(
            "--qdrant-url",
            default=None,
            help="Qdrant URL (defaults to env QDRANT_URL).",
        )
        parser.add_argument(
            "--qdrant-collection",
            default=None,
            help="Qdrant collection (defaults to env QDRANT_COLLECTION).",
        )
    return parser


def add_topic_args(parser: argparse.ArgumentParser, *, include_k_sweep: bool) -> None:
    parser.add_argument("--topic-n-topics", type=int, default=None)
    if include_k_sweep:
        parser.add_argument("--topic-k-multiplier", action="append", type=float, default=None)
        parser.add_argument("--topic-k-value", action="append", type=int, default=None)
    parser.add_argument("--topic-max-features", type=int, default=None)
    parser.add_argument("--topic-min-df", type=float, default=None)
    parser.add_argument("--topic-max-df", type=float, default=None)
    parser.add_argument("--topic-random-state", type=int, default=None)
    parser.add_argument("--bertopic-semisupervised-label-fraction", type=float, default=None)


def settings_from_family_args(
    args: argparse.Namespace,
    *,
    family: str,
) -> base.Settings:
    app_settings = AppSettings.from_env()
    canonical_family = {"simple": "zero_shot", "topic": "unsupervised"}.get(family, family)
    settings = base.Settings(
        strategy_name=app_settings.strategy_name,
        mongo_uri=app_settings.mongo_uri,
        mongo_db_name=app_settings.mongo_db_name,
        qdrant_url=app_settings.qdrant_url,
        qdrant_collection=app_settings.qdrant_collection,
        baseline_kinds=FAMILY_BASELINES[family],
        base_output_dir=f"outputs/baselines/{canonical_family}",
    )
    data: dict[str, Any] = dataclasses.asdict(settings)
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
        "supervised_frozen_model",
        "supervised_frozen_text_source",
        "supervised_frozen_pooling",
        "supervised_frozen_max_length",
        "supervised_frozen_batch_size",
        "supervised_frozen_cache_dir",
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "random_chunk_k",
        "random_chunk_seed",
        "qdrant_url",
        "qdrant_collection",
        "nli_model",
        "nli_hypothesis_source",
        "nli_hypothesis_template",
        "nli_threshold",
        "nli_max_length",
        "nli_batch_size",
        "nli_cache_dir",
    ):
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                data[field] = value
    if hasattr(args, "topic_k_multiplier") and args.topic_k_multiplier is not None:
        data["topic_k_multipliers"] = tuple(args.topic_k_multiplier)
    if hasattr(args, "topic_k_value") and args.topic_k_value is not None:
        data["topic_k_values"] = tuple(args.topic_k_value)
    if (
        hasattr(args, "topic_n_topics")
        and args.topic_n_topics is not None
        and (not hasattr(args, "topic_k_multiplier") or args.topic_k_multiplier is None)
        and (not hasattr(args, "topic_k_value") or args.topic_k_value is None)
    ):
        data["topic_k_multipliers"] = ()
        data["topic_k_values"] = ()
    if hasattr(args, "supervised_cv_mode") and args.supervised_cv_mode is not None:
        data["supervised_cv_modes"] = tuple(args.supervised_cv_mode)
    if getattr(args, "supervised_frozen_no_normalize", False):
        data["supervised_frozen_normalize"] = False
    if args.overwrite:
        data["overwrite"] = True
    data["classifier_kinds"] = base.expand_arg_values(
        args.classifier_kind,
        all_values=base.CLASSIFIER_KINDS,
    )
    if args.baseline_kind is not None:
        data["baseline_kinds"] = expand_family_baselines(args.baseline_kind, family=family)
    return base.Settings(**data)


def expand_family_baselines(values: Sequence[str], *, family: str) -> tuple[str, ...]:
    out = []
    for value in values:
        if value == "all":
            out.extend(FAMILY_BASELINES[family])
        else:
            normalized = base.normalize_baseline_kind(value)
            if normalized not in FAMILY_BASELINES[family]:
                raise ValueError(
                    f"Baseline {value!r} is not part of the {family!r} family."
                )
            out.append(normalized)
    deduped = []
    for value in out:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def run_family(argv: Sequence[str] | None = None, *, family: str, description: str) -> None:
    parser = family_parser(family=family, description=description)
    settings = settings_from_family_args(parser.parse_args(argv), family=family)
    records = base.load_records(settings.input_csv, settings.input_csv_sep)
    ground_truth_records = (
        base.load_records(settings.ground_truth_csv, settings.ground_truth_csv_sep)
        if settings.ground_truth_csv
        else records
    )
    failures = 0
    for classifier_kind in settings.classifier_kinds:
        for baseline_kind in settings.baseline_kinds:
            for run_settings in base.settings_for_baseline(
                settings,
                classifier_kind=classifier_kind,
                baseline_kind=baseline_kind,
            ):
                for repeat_idx in range(1, run_settings.repeats + 1):
                    try:
                        print(
                            f"\n=== task={classifier_kind} baseline={baseline_kind} "
                            f"repeat={repeat_idx}/{run_settings.repeats} ==="
                        )
                        base.run_once(
                            run_settings,
                            classifier_kind=classifier_kind,
                            baseline_kind=baseline_kind,
                            repeat_idx=repeat_idx,
                            records=records,
                            ground_truth_records=ground_truth_records,
                        )
                    except Exception as exc:
                        failures += 1
                        print(f"[ERROR] task={classifier_kind} baseline={baseline_kind}: {exc!r}")
    if failures:
        raise SystemExit(1)
