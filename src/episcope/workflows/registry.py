"""Single source of truth for classifier and precision-miner *kinds*.

A "kind" is the short string that selects which workflow configuration to run
(e.g. ``data_accessibility`` or ``find_data_sources``). Built-in kinds live in
:data:`CLASSIFIERS` / :data:`MINERS`; the CLI, API, runtime, and UI all resolve
kinds through this module, and the API publishes the catalog on ``/health`` so
the UI never hardcodes anything.

User-defined tasks (config-as-data)
-----------------------------------
Tasks can also be declared as plain JSON via :class:`TaskSpec` and registered at
runtime, so non-experts can add their own classifier or miner without writing
Python. Two delivery paths are supported:

* **Workspace tasks** — JSON files under ``<workspace>/tasks/*.json`` that the
  CLI auto-discovers (and a server can load via ``EPISCOPE_TASKS_DIR``).
* **Portable / inline specs** — a single spec file passed to the CLI
  (``--task-file``) or sent inline in an API request, so the *same* definition
  works across entrypoints without server-side state.

A classifier task supplies a label set (codes, names, definitions, optional
retrieval example sentences); valid codes are injected into the prompt and
enforced when parsing. A miner task supplies retrieval templates and prompts
(the extraction schema is fixed). Prompts default to sensible templates and are
``str.format`` strings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from episcope.workflows.classification.config import (
    BaseClassifierConfig,
    DataAccessibilityClassifierConfig,
    DataTypeClassifierConfig,
    GeoClassifierConfig,
    PaperTypeClassifierConfig,
)
from episcope.workflows.classification.schemas import BaseClassificationSchema
from episcope.workflows.precision_miner.config import (
    FindDataSourcesConfig,
    FindSupplementaryLinksConfig,
    IdentifyKeyReferencesConfig,
    PrecisionMinerConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowKind:
    """One selectable workflow variant (a classifier or a miner).

    ``factory`` is a zero-argument callable (a config class for built-ins, or a
    spec-bound builder for declarative tasks) that returns a fresh config.
    ``source`` is ``"builtin"`` or ``"declarative"``.
    """

    key: str
    label: str
    factory: Callable[[], Any]
    description: str = ""
    source: str = "builtin"


CLASSIFIERS: Dict[str, WorkflowKind] = {
    spec.key: spec
    for spec in (
        WorkflowKind(
            "paper_type",
            "Paper type",
            PaperTypeClassifierConfig,
            "Contribution type to epidemiological parameter estimation.",
        ),
        WorkflowKind(
            "data_accessibility",
            "Data availability",
            DataAccessibilityClassifierConfig,
            "How accessible the dataset(s) used in the paper are.",
        ),
        WorkflowKind(
            "data_type",
            "Data type",
            DataTypeClassifierConfig,
            "Type(s) of data used as evidence (traditional vs non-traditional).",
        ),
        WorkflowKind(
            "geo",
            "Geographic scope",
            GeoClassifierConfig,
            "Continent-level geography tied to the evidence.",
        ),
    )
}


MINERS: Dict[str, WorkflowKind] = {
    spec.key: spec
    for spec in (
        WorkflowKind(
            "find_data_sources",
            "Find data sources",
            FindDataSourcesConfig,
            "Extract datasets, databases, and repositories used in the study.",
        ),
        WorkflowKind(
            "find_supplementary_links",
            "Find supplementary links",
            FindSupplementaryLinksConfig,
            "Extract supplementary materials, appendices, and their links.",
        ),
        WorkflowKind(
            "identify_key_references",
            "Identify key references",
            IdentifyKeyReferencesConfig,
            "Extract foundational references central to the study.",
        ),
    )
}

# Captured before any declarative task can be registered; built-in keys are
# protected from being overridden by user tasks.
BUILTIN_CLASSIFIER_KEYS = frozenset(CLASSIFIERS)
BUILTIN_MINER_KEYS = frozenset(MINERS)


# ---------------------------------------------------------------------------
# Resolution / dispatch
# ---------------------------------------------------------------------------
def _normalize(kind: object) -> str:
    return str(getattr(kind, "value", kind))


def build_classifier_config(kind: object, top_k: int) -> "BaseClassifierConfig":
    """Instantiate the classifier config for ``kind`` and apply ``top_k``."""
    key = _normalize(kind)
    spec = CLASSIFIERS.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown classifier kind {key!r}. Available: {', '.join(CLASSIFIERS)}."
        )
    config = spec.factory()
    config.top_k = top_k
    return config


def build_precision_miner_config(kind: object, top_k: int) -> "PrecisionMinerConfig":
    """Instantiate the precision-miner config for ``kind`` and apply ``top_k``."""
    key = _normalize(kind)
    spec = MINERS.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown precision-miner kind {key!r}. Available: {', '.join(MINERS)}."
        )
    config = spec.factory()
    config.top_k = top_k
    return config


def classifier_catalog() -> List[Dict[str, str]]:
    """Serializable ``[{key, label, description, source}]`` for clients."""
    return [_catalog_entry(spec) for spec in CLASSIFIERS.values()]


def miner_catalog() -> List[Dict[str, str]]:
    """Serializable ``[{key, label, description, source}]`` for clients."""
    return [_catalog_entry(spec) for spec in MINERS.values()]


def _catalog_entry(spec: WorkflowKind) -> Dict[str, str]:
    return {
        "key": spec.key,
        "label": spec.label,
        "description": spec.description,
        "source": spec.source,
    }


# ---------------------------------------------------------------------------
# Declarative tasks (config-as-data)
# ---------------------------------------------------------------------------
class LabelSpec(BaseModel):
    """One label in a declarative classifier task."""

    code: str = Field(..., description="Value the model emits and that appears in results.")
    name: str = Field("", description="Human-readable label for display.")
    definition: str = Field("", description="Definition injected into the prompt.")
    examples: List[str] = Field(
        default_factory=list,
        description="Prototypical sentences used to seed retrieval for this label.",
    )

    @field_validator("code")
    @classmethod
    def _non_empty_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label code must be non-empty")
        return value


class TaskSpec(BaseModel):
    """A user-defined classifier or miner task, expressed as data."""

    key: str = Field(..., description="Unique kind name (a-z, 0-9, underscore).")
    kind: Literal["classifier", "miner"]
    label: str = Field("", description="Human-readable name; defaults from key.")
    description: str = ""
    top_k: Optional[int] = Field(None, description="Default retrieval depth.")

    # Classifier fields
    labels: List[LabelSpec] = Field(default_factory=list)
    multi_label: bool = True
    default_label: Optional[str] = Field(
        None, description="Fallback label code when classification fails."
    )

    # Miner fields
    retrieval_templates: List[str] = Field(default_factory=list)
    section_filters: Optional[List[str]] = None

    # Optional prompt overrides (str.format templates)
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", value):
            raise ValueError(
                "key must contain only lowercase letters, digits, and underscores"
            )
        return value

    @model_validator(mode="after")
    def _validate(self) -> "TaskSpec":
        if not self.label:
            self.label = self.key.replace("_", " ").title()
        if self.kind == "classifier":
            if not self.labels:
                raise ValueError(
                    f"classifier task {self.key!r} must define at least one label"
                )
            codes = [label.code for label in self.labels]
            if len(set(codes)) != len(codes):
                raise ValueError(
                    f"classifier task {self.key!r} has duplicate label codes"
                )
            if self.default_label is not None and self.default_label not in codes:
                raise ValueError(
                    f"default_label {self.default_label!r} is not one of the label codes"
                )
        return self


DEFAULT_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a careful research assistant. Assign labels to a scientific paper "
    "based only on the provided text."
)

DEFAULT_MINER_SYSTEM_PROMPT = (
    "You are a careful research assistant extracting structured items from a "
    "scientific paper based only on the provided text."
)

DEFAULT_MINER_USER_TEMPLATE = (
    "Extract the requested items from the paper.\n\n"
    "**Relevant Extracts:**\n{chunks_info}\n\n"
    "**Instructions:**\n"
    "1. Use only the evidence above.\n"
    "2. For each item, provide a name, a URL if available, and a short explanation.\n"
    "3. Return a single JSON object matching the schema. Do not include extra text.\n\n"
    "**Schema:**\n{schema}\n"
)


def _classifier_user_template(multi_label: bool) -> str:
    choose = (
        "all applicable categories" if multi_label else "exactly one category"
    )
    return (
        "You are classifying a scientific paper.\n\n"
        f"**Categories (choose {choose} by code):**\n{{categories}}\n\n"
        "**Definitions:**\n{definitions}\n\n"
        "**Paper Content:**\n"
        "Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}\n\n"
        "**Relevant Extracts (candidate evidence):**\n{chunks_info}\n\n"
        "**Instructions:**\n"
        "1. Base your decision only on the provided text.\n"
        "2. Populate `classification` with the chosen category code(s).\n"
        "3. Return a single JSON object matching the schema below. "
        "Do not include any extra text.\n\n"
        "**Schema:**\n{schema}\n"
    )


class DeclarativeClassificationOutput(BaseClassificationSchema):
    """Generic output schema shared by all declarative classifier tasks.

    Valid codes are conveyed to the model via the prompt; unknown codes are
    dropped (in favour of the task's default label) when the response is parsed.
    """

    classification: List[str] = Field(
        default_factory=list, description="Selected category code(s)."
    )
    primary_label: Optional[str] = Field(
        default=None, description="Single dominant code, or null."
    )
    extras: Dict[str, Any] = Field(
        default_factory=dict, description="Optional free-form extracted fields."
    )


def build_classifier_config_from_spec(spec: TaskSpec) -> BaseClassifierConfig:
    """Build a runnable classifier config from a declarative :class:`TaskSpec`."""
    if spec.kind != "classifier":
        raise ValueError(f"Task {spec.key!r} is not a classifier task.")
    codes = [label.code for label in spec.labels]
    category_labels = {label.code: (label.name or label.code) for label in spec.labels}
    classification_mapping = {label.code: label.code for label in spec.labels}
    template_paragraphs = {
        label.code: list(label.examples) for label in spec.labels if label.examples
    }
    definitions = "\n".join(
        f"- {label.code}: {label.definition}"
        for label in spec.labels
        if label.definition
    )
    return BaseClassifierConfig(
        top_k=spec.top_k or 10,
        template_paragraphs=template_paragraphs,
        classification_mapping=classification_mapping,
        category_labels=category_labels,
        system_prompt=spec.system_prompt or DEFAULT_CLASSIFIER_SYSTEM_PROMPT,
        user_prompt_template=(
            spec.user_prompt_template or _classifier_user_template(spec.multi_label)
        ),
        output_schema=DeclarativeClassificationOutput,
        default_classification=[spec.default_label or codes[-1]],
        multi_label=spec.multi_label,
        extra_output_fields={"definitions": definitions},
    )


def build_miner_config_from_spec(spec: TaskSpec) -> PrecisionMinerConfig:
    """Build a runnable miner config from a declarative :class:`TaskSpec`."""
    if spec.kind != "miner":
        raise ValueError(f"Task {spec.key!r} is not a miner task.")
    return PrecisionMinerConfig(
        top_k=spec.top_k or 15,
        retrieval_templates=list(spec.retrieval_templates),
        section_filters=spec.section_filters,
        system_prompt=spec.system_prompt or DEFAULT_MINER_SYSTEM_PROMPT,
        user_prompt_template=spec.user_prompt_template or DEFAULT_MINER_USER_TEMPLATE,
    )


def register_task(spec: TaskSpec, *, overwrite: bool = False) -> str:
    """Register a declarative task into the live registry; returns its key."""
    if spec.key in BUILTIN_CLASSIFIER_KEYS or spec.key in BUILTIN_MINER_KEYS:
        raise ValueError(
            f"Task key {spec.key!r} is a built-in kind and cannot be overridden."
        )
    target = CLASSIFIERS if spec.kind == "classifier" else MINERS
    other = MINERS if spec.kind == "classifier" else CLASSIFIERS
    if spec.key in other:
        raise ValueError(
            f"Task key {spec.key!r} is already used by a different workflow family."
        )
    if spec.key in target and not overwrite:
        raise ValueError(
            f"Task {spec.key!r} is already registered (pass overwrite=True to replace)."
        )
    if spec.kind == "classifier":
        factory: Callable[[], Any] = partial(build_classifier_config_from_spec, spec)
    else:
        factory = partial(build_miner_config_from_spec, spec)
    target[spec.key] = WorkflowKind(
        key=spec.key,
        label=spec.label,
        factory=factory,
        description=spec.description,
        source="declarative",
    )
    return spec.key


def validate_task_file(path: str | Path) -> TaskSpec:
    """Parse and validate a JSON task spec file *without* registering it.

    Raises ``ValueError`` / ``pydantic.ValidationError`` with a clear message
    if the spec is malformed. Intended for ``episcope tasks validate``.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return TaskSpec.model_validate(data)


def scaffold_task_spec(kind: str) -> str:
    """Return a filled-in JSON scaffold for a new task spec.

    ``kind`` must be ``"classifier"`` or ``"miner"``.  Every field is
    present so the author can see what is available; optional fields are
    set to representative placeholder values rather than omitted.
    """
    if kind == "classifier":
        spec: Dict[str, Any] = {
            "key": "my_classifier",
            "kind": "classifier",
            "label": "My classifier",
            "description": "What this classifier detects.",
            "top_k": 10,
            "multi_label": True,
            "default_label": "unclear",
            "system_prompt": None,
            "user_prompt_template": None,
            "labels": [
                {
                    "code": "label_a",
                    "name": "Label A",
                    "definition": "Definition of label A.",
                    "examples": [
                        "A representative sentence that should retrieve label_a evidence."
                    ],
                },
                {
                    "code": "label_b",
                    "name": "Label B",
                    "definition": "Definition of label B.",
                    "examples": [],
                },
                {
                    "code": "unclear",
                    "name": "Unclear",
                    "definition": "Not enough information to classify.",
                    "examples": [],
                },
            ],
        }
    elif kind == "miner":
        spec = {
            "key": "my_miner",
            "kind": "miner",
            "label": "My miner",
            "description": "What this miner extracts.",
            "top_k": 15,
            "section_filters": None,
            "system_prompt": None,
            "user_prompt_template": None,
            "retrieval_templates": [
                "First retrieval question to find relevant evidence.",
                "Second retrieval question using different wording.",
            ],
        }
    else:
        raise ValueError(
            f"Unknown task kind {kind!r}. Choose 'classifier' or 'miner'."
        )
    return json.dumps(spec, indent=2, ensure_ascii=False)


def load_task_file(path: str | Path, *, overwrite: bool = False) -> TaskSpec:
    """Load, validate, and register a JSON task spec; returns the parsed spec."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = TaskSpec.model_validate(data)
    register_task(spec, overwrite=overwrite)
    return spec


def load_tasks_dir(path: str | Path, *, overwrite: bool = True) -> List[TaskSpec]:
    """Register every ``*.json`` task spec in a directory (no-op if absent)."""
    directory = Path(path)
    if not directory.is_dir():
        return []
    specs: List[TaskSpec] = []
    for task_file in sorted(directory.glob("*.json")):
        specs.append(load_task_file(task_file, overwrite=overwrite))
    return specs


def _load_startup_tasks() -> None:
    tasks_dir = os.environ.get("EPISCOPE_TASKS_DIR")
    if not tasks_dir:
        return
    try:
        loaded = load_tasks_dir(tasks_dir)
        if loaded:
            logger.info(
                "Loaded %d task(s) from EPISCOPE_TASKS_DIR=%s", len(loaded), tasks_dir
            )
    except Exception as exc:  # best-effort: a bad tasks dir must not break imports
        logger.warning("Failed to load EPISCOPE_TASKS_DIR=%s: %s", tasks_dir, exc)


_load_startup_tasks()
