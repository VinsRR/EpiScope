"""Schema and loader for classification experiment configs.

An experiment config is a JSON object describing one "named" run of the
classification baseline grid. It encodes which baselines to run, with
which per-baseline overrides, against which evaluation sample, and where
the results should land. The same config is consumed by:

  - ``classification.experiments.run`` to dispatch the family runners,
  - ``eval.experiments.report`` to evaluate the resulting TSVs and emit
    paper tables / figures / significance results.

Both halves share a single source of truth, which is the point of this
layer: every figure, table, and significance row in the paper can be
traced back to one config file.

Schema (all paths are resolved relative to the repository root unless
absolute):

    {
      "name": "paper_main",
      "description": "...",
      "ground_truth_csv": "sampled_papers_full.csv",
      "tasks": ["paper_type", "data_accessibility", "data_type", "geo"],
      "default_text_scope": "full_text",
      "default_repeats": 1,
      "random_state": 13,
      "output_root": "outputs/baselines",
      "eval_root": "eval_outputs/classification/paper_main",
      "baselines": [
        {
          "id": "majority",
          "label": "Majority class",
          "family": "zero_shot",          # zero_shot | supervised | frozen | llm | external
          "kind": "majority",             # baseline_kind for the family runner
          "paper_role": "main",           # main | appendix | ablation
          "args": { ... },                # CLI flag overrides for the family runner
          "external_run_root": null       # set instead of "kind" for family=external
        },
        ...
      ],
      "significance_pairs": [             # optional; pair ids resolved to baselines above
        {"a": "episcope", "b": "metadata_llm"},
        {"a": "episcope", "b": "tfidf_logreg"}
      ]
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from eval.common.config import load_json_config


VALID_FAMILIES = {"zero_shot", "supervised", "frozen", "llm", "external"}
VALID_ROLES = {"main", "appendix", "ablation"}
DEFAULT_TASKS = ("paper_type", "data_accessibility", "data_type", "geo")


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline within an experiment config."""

    id: str
    label: str
    family: str
    kind: Optional[str] = None
    paper_role: str = "main"
    args: Dict[str, Any] = field(default_factory=dict)
    # For ``family="external"``: one or more paths to existing prediction
    # directories. Each entry may itself be a glob pattern (containing ``*``
    # or ``?``); the report expands globs against the repository root so a
    # single baseline can pull together fan-out layouts like
    # ``outputs/baselines/zero_shot/*/majority``.
    external_run_root: tuple[str, ...] = ()
    # Populated by the runner after the family CLI executes (or, for external
    # baselines, derived from ``external_run_root`` at config-load time).
    run_root_default: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in VALID_FAMILIES:
            raise ValueError(
                f"Baseline {self.id!r}: family must be one of {sorted(VALID_FAMILIES)}; "
                f"got {self.family!r}."
            )
        if self.paper_role not in VALID_ROLES:
            raise ValueError(
                f"Baseline {self.id!r}: paper_role must be one of {sorted(VALID_ROLES)}; "
                f"got {self.paper_role!r}."
            )
        if self.family == "external":
            if not self.external_run_root:
                raise ValueError(
                    f"Baseline {self.id!r}: family='external' requires "
                    "'external_run_root' to be set (string or list of strings)."
                )
            if self.kind is not None:
                raise ValueError(
                    f"Baseline {self.id!r}: family='external' must not set 'kind'."
                )
        else:
            if not self.kind:
                raise ValueError(
                    f"Baseline {self.id!r}: family={self.family!r} requires 'kind'."
                )


@dataclass(frozen=True)
class SignificancePair:
    a: str
    b: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment description."""

    name: str
    path: Path
    ground_truth_csv: str
    tasks: tuple[str, ...]
    default_text_scope: str
    default_repeats: int
    random_state: int
    output_root: str
    eval_root: str
    baselines: tuple[BaselineSpec, ...]
    significance_pairs: tuple[SignificancePair, ...] = ()
    description: str = ""

    def baseline_by_id(self, baseline_id: str) -> BaselineSpec:
        for spec in self.baselines:
            if spec.id == baseline_id:
                return spec
        raise KeyError(
            f"Experiment {self.name!r}: no baseline with id={baseline_id!r}. "
            f"Available: {sorted(s.id for s in self.baselines)}"
        )

    def baselines_by_role(self, role: str) -> tuple[BaselineSpec, ...]:
        return tuple(b for b in self.baselines if b.paper_role == role)


ALLOWED_TOP_KEYS = {
    "name",
    "description",
    "ground_truth_csv",
    "tasks",
    "default_text_scope",
    "default_repeats",
    "random_state",
    "output_root",
    "eval_root",
    "baselines",
    "significance_pairs",
}

ALLOWED_BASELINE_KEYS = {
    "id",
    "label",
    "family",
    "kind",
    "paper_role",
    "args",
    "external_run_root",
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a classification experiment config from a JSON file."""
    config_path = Path(path).resolve()
    data = load_json_config(config_path, allowed_keys=ALLOWED_TOP_KEYS)

    name = str(data.get("name") or config_path.stem)
    ground_truth = str(data.get("ground_truth_csv") or "sampled_papers_full.csv")
    tasks = tuple(data.get("tasks") or DEFAULT_TASKS)
    default_text_scope = str(data.get("default_text_scope") or "full_text")
    default_repeats = int(data.get("default_repeats") or 1)
    random_state = int(data.get("random_state") or 13)
    output_root = str(data.get("output_root") or "outputs/baselines")
    eval_root = str(data.get("eval_root") or f"eval_outputs/classification/{name}")
    description = str(data.get("description") or "")

    raw_baselines = data.get("baselines") or []
    if not isinstance(raw_baselines, list) or not raw_baselines:
        raise ValueError(
            f"Experiment config {config_path} must list at least one baseline "
            "under 'baselines'."
        )

    baselines: list[BaselineSpec] = []
    seen_ids: set[str] = set()
    for entry in raw_baselines:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Experiment config {config_path}: each baseline must be a JSON "
                f"object; got {type(entry).__name__}."
            )
        unknown = sorted(set(entry) - ALLOWED_BASELINE_KEYS)
        if unknown:
            raise ValueError(
                f"Experiment config {config_path}: unknown baseline keys "
                f"{unknown}. Allowed: {sorted(ALLOWED_BASELINE_KEYS)}."
            )
        spec_id = str(entry.get("id") or "")
        if not spec_id:
            raise ValueError(f"Experiment config {config_path}: baseline missing 'id'.")
        if spec_id in seen_ids:
            raise ValueError(
                f"Experiment config {config_path}: duplicate baseline id={spec_id!r}."
            )
        seen_ids.add(spec_id)
        raw_external = entry.get("external_run_root")
        if raw_external is None or raw_external == "":
            external_paths: tuple[str, ...] = ()
        elif isinstance(raw_external, str):
            external_paths = (raw_external.strip(),)
        elif isinstance(raw_external, (list, tuple)):
            external_paths = tuple(str(p).strip() for p in raw_external if str(p).strip())
        else:
            raise ValueError(
                f"Baseline {spec_id!r}: external_run_root must be a string or "
                f"list of strings; got {type(raw_external).__name__}."
            )
        baselines.append(
            BaselineSpec(
                id=spec_id,
                label=str(entry.get("label") or spec_id),
                family=str(entry.get("family") or "zero_shot"),
                kind=entry.get("kind"),
                paper_role=str(entry.get("paper_role") or "main"),
                args=dict(entry.get("args") or {}),
                external_run_root=external_paths,
            )
        )

    raw_pairs = data.get("significance_pairs") or []
    pairs: list[SignificancePair] = []
    for entry in raw_pairs:
        if not isinstance(entry, dict) or "a" not in entry or "b" not in entry:
            raise ValueError(
                f"Experiment config {config_path}: each significance pair must be "
                "an object with 'a' and 'b' keys."
            )
        pair = SignificancePair(a=str(entry["a"]), b=str(entry["b"]))
        for side in (pair.a, pair.b):
            if side not in seen_ids:
                raise ValueError(
                    f"Experiment config {config_path}: significance pair references "
                    f"unknown baseline id={side!r}."
                )
        pairs.append(pair)

    return ExperimentConfig(
        name=name,
        path=config_path,
        ground_truth_csv=ground_truth,
        tasks=tasks,
        default_text_scope=default_text_scope,
        default_repeats=default_repeats,
        random_state=random_state,
        output_root=output_root,
        eval_root=eval_root,
        baselines=tuple(baselines),
        significance_pairs=tuple(pairs),
        description=description,
    )
