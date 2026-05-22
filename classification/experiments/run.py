"""Run all baselines listed in an experiment config.

Reads one experiment config (see ``classification.experiments.config``),
dispatches each baseline to the appropriate family runner with the
config's per-baseline overrides, and writes a manifest under
``<eval_root>/manifest.json`` mapping baseline ids to their expected
output directories. The manifest is what ``eval.experiments.report``
consumes downstream — it lets the eval step locate predictions without
re-deriving paths from the family's slug logic.

Usage
-----

    python -m classification.experiments.run \\
        --config eval/configs/classification/paper_main.json

    python -m classification.experiments.run \\
        --config eval/configs/classification/paper_main.json \\
        --only majority --only tfidf_logreg

    python -m classification.experiments.run \\
        --config eval/configs/classification/paper_main.json \\
        --dry-run

Existing per-script CLIs (``scripts/run_*.py``) continue to work
unchanged — this runner is an addition that composes them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

# Set parallelism guards before any HF/UMAP import to match the standalone
# scripts under scripts/.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from classification.experiments.config import (
    BaselineSpec,
    ExperimentConfig,
    load_experiment_config,
)


# Map family name -> runner module providing main(argv).
FAMILY_RUNNERS = {
    "zero_shot": "classification.runners.zero_shot",
    "supervised": "classification.runners.supervised",
    "frozen": "classification.runners.frozen",
    "llm": "classification.runners.llm",
}


def _format_arg_value(value: Any) -> List[str]:
    """Render a config arg value into argv tokens.

    - bool True -> just the flag (store_true semantics).
    - bool False -> dropped (we assume default-False store_true flags).
    - list/tuple -> the flag is repeated, one occurrence per value.
    - other scalars -> single value.
    """
    if isinstance(value, bool):
        return [] if not value else [""]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def baseline_to_argv(
    spec: BaselineSpec,
    *,
    experiment: ExperimentConfig,
) -> List[str]:
    """Translate a baseline spec into the argv expected by its family runner."""
    argv: List[str] = [
        "--baseline-kind",
        spec.kind,
        "--ground-truth-csv",
        experiment.ground_truth_csv,
        "--input-csv",
        experiment.ground_truth_csv,
        "--base-output-dir",
        str(_baseline_output_dir(spec, experiment)),
        "--text-scope",
        experiment.default_text_scope,
        "--repeats",
        str(experiment.default_repeats),
    ]
    # Each task is added separately; the family runner accepts --classifier-kind
    # as ``action="append"``.
    for task in experiment.tasks:
        argv.extend(["--classifier-kind", task])

    # Per-baseline overrides win over the experiment-level defaults above.
    # If the same flag is set in args, we replace the earlier occurrence.
    overrides = dict(spec.args)
    if overrides:
        argv = _apply_overrides(argv, overrides)
    return argv


def _baseline_output_dir(spec: BaselineSpec, experiment: ExperimentConfig) -> Path:
    """Where this baseline's family runner should write its outputs.

    The family runner appends its own ``<task>/<method-slug>/<strategy>/...``
    structure underneath this directory, so per-baseline isolation lets the
    manifest unambiguously point at each baseline's outputs.
    """
    return Path(experiment.output_root) / experiment.name / spec.id


def _apply_overrides(argv: List[str], overrides: dict[str, Any]) -> List[str]:
    """Drop any pre-existing occurrence of an overridden flag, then append it.

    This matches argparse's "last wins" behaviour for non-append actions
    while letting list-valued overrides expand into repeated flags.
    """
    flag_prefixes = {f"--{flag}" for flag in overrides}
    cleaned: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in flag_prefixes:
            skip_next = True
            continue
        cleaned.append(token)
    for flag, value in overrides.items():
        if isinstance(value, bool):
            if value:
                cleaned.append(f"--{flag}")
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                cleaned.extend([f"--{flag}", str(item)])
        else:
            cleaned.extend([f"--{flag}", str(value)])
    return cleaned


def _load_family_main(family: str):
    import importlib

    if family not in FAMILY_RUNNERS:
        raise ValueError(
            f"family={family!r} is not dispatchable. Use one of "
            f"{sorted(FAMILY_RUNNERS)}, or set family='external' and provide "
            "'external_run_root'."
        )
    module = importlib.import_module(FAMILY_RUNNERS[family])
    if not hasattr(module, "main"):
        raise RuntimeError(
            f"Family runner {FAMILY_RUNNERS[family]} does not expose main(argv)."
        )
    return module.main


def write_manifest(
    experiment: ExperimentConfig,
    *,
    statuses: dict[str, dict[str, Any]],
) -> Path:
    eval_root = Path(experiment.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_root / "manifest.json"
    manifest = {
        "experiment_name": experiment.name,
        "config_path": str(experiment.path),
        "description": experiment.description,
        "ground_truth_csv": experiment.ground_truth_csv,
        "tasks": list(experiment.tasks),
        "default_text_scope": experiment.default_text_scope,
        "default_repeats": experiment.default_repeats,
        "random_state": experiment.random_state,
        "output_root": experiment.output_root,
        "eval_root": experiment.eval_root,
        "baselines": [],
        "significance_pairs": [
            {"a": pair.a, "b": pair.b} for pair in experiment.significance_pairs
        ],
    }
    for spec in experiment.baselines:
        entry: dict[str, Any] = {
            "id": spec.id,
            "label": spec.label,
            "family": spec.family,
            "kind": spec.kind,
            "paper_role": spec.paper_role,
            "args": dict(spec.args),
        }
        if spec.family == "external":
            # Preserve list shape for downstream globbing; single-entry lists
            # are written as their lone string for readability.
            external = list(spec.external_run_root)
            entry["run_root"] = external[0] if len(external) == 1 else external
            entry["status"] = "external"
        else:
            entry["run_root"] = str(_baseline_output_dir(spec, experiment))
            entry["status"] = statuses.get(spec.id, {}).get("status", "pending")
            duration = statuses.get(spec.id, {}).get("duration_s")
            if duration is not None:
                entry["duration_s"] = duration
            error = statuses.get(spec.id, {}).get("error")
            if error is not None:
                entry["error"] = error
        manifest["baselines"].append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_experiment(
    config_path: str | Path,
    *,
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> int:
    """Run every (selected) baseline in the experiment config.

    Returns the number of baselines that failed (0 means success).
    """
    experiment = load_experiment_config(config_path)
    only_set = {x for x in (only or [])}
    skip_set = {x for x in (skip or [])}
    if only_set:
        unknown = only_set - {b.id for b in experiment.baselines}
        if unknown:
            raise ValueError(f"--only references unknown baseline ids: {sorted(unknown)}")

    Path(experiment.eval_root).mkdir(parents=True, exist_ok=True)
    print(f"=== experiment {experiment.name!r} ({experiment.path}) ===")
    print(f"    tasks={list(experiment.tasks)}  text_scope={experiment.default_text_scope}  "
          f"repeats={experiment.default_repeats}  random_state={experiment.random_state}")
    print(f"    output_root={experiment.output_root}  eval_root={experiment.eval_root}")

    statuses: dict[str, dict[str, Any]] = {}
    failures = 0

    for spec in experiment.baselines:
        if only_set and spec.id not in only_set:
            continue
        if spec.id in skip_set:
            continue
        if spec.family == "external":
            print(f"\n--- {spec.id} ({spec.label})  [external run root: {spec.external_run_root}]")
            continue

        argv = baseline_to_argv(spec, experiment=experiment)
        print(f"\n--- {spec.id} ({spec.label})")
        print(f"    family={spec.family} kind={spec.kind}")
        print(f"    argv: {' '.join(argv)}")
        if dry_run:
            statuses[spec.id] = {"status": "dry-run"}
            continue

        family_main = _load_family_main(spec.family)
        start = time.time()
        try:
            family_main(argv)
            statuses[spec.id] = {"status": "ok", "duration_s": round(time.time() - start, 2)}
        except SystemExit as exc:
            duration = round(time.time() - start, 2)
            code = exc.code if isinstance(exc.code, int) else 1
            if code == 0:
                statuses[spec.id] = {"status": "ok", "duration_s": duration}
            else:
                failures += 1
                statuses[spec.id] = {
                    "status": "failed",
                    "duration_s": duration,
                    "error": f"SystemExit({code})",
                }
                print(f"[ERROR] baseline {spec.id} exited with code {code}.")
        except Exception as exc:
            failures += 1
            statuses[spec.id] = {
                "status": "failed",
                "duration_s": round(time.time() - start, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[ERROR] baseline {spec.id} raised: {exc!r}")
            traceback.print_exc()

    manifest_path = write_manifest(experiment, statuses=statuses)
    print(f"\nWrote manifest to {manifest_path}")
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run classification baselines from an experiment config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON experiment config (e.g. eval/configs/classification/paper_main.json).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named baseline id (repeatable).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip the named baseline id (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned per-baseline argv invocations without executing them.",
    )
    args = parser.parse_args(argv)
    failures = run_experiment(
        args.config,
        only=args.only or None,
        skip=args.skip or None,
        dry_run=args.dry_run,
    )
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
