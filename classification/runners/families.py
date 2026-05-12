from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from typing import Any

from episcope.settings import AppSettings

import scripts.run_classification_baselines as base


FAMILY_BASELINES = {
    "simple": ("majority", "prototype_similarity"),
    "topic": base.TOPIC_MODEL_BASELINES,
    "supervised": base.SUPERVISED_BASELINES,
    "llm": ("metadata_llm",),
}


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
    if family == "simple":
        parser.add_argument("--majority-fit-mode", choices=["leave_one_out", "all"], default=None)
        parser.add_argument("--prototype-multilabel-ratio", type=float, default=None)
        parser.add_argument("--prototype-min-score", type=float, default=None)
    if family == "topic":
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
    if family == "llm":
        parser.add_argument(
            "--llm-provider",
            choices=["gemini", "openrouter", "openai", "ollama"],
            default=None,
        )
        parser.add_argument("--llm-model", default=None)
        parser.add_argument("--llm-temperature", type=float, default=None)
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
    settings = base.Settings(
        strategy_name=app_settings.strategy_name,
        mongo_uri=app_settings.mongo_uri,
        mongo_db_name=app_settings.mongo_db_name,
        baseline_kinds=FAMILY_BASELINES[family],
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
