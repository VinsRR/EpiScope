"""Single source of truth for classifier and precision-miner *kinds*.

A "kind" is the short string that selects which workflow configuration to run
(e.g. ``data_accessibility`` or ``find_data_sources``). This set used to be
duplicated across the CLI (a Typer ``Enum``), the API (a Pydantic ``Literal``),
the runtime (dispatch ``dict``s), and the Streamlit UI (hardcoded lists), and it
had already drifted.

Everything now derives from the two registries below:

* the CLI and API share the generated :data:`ClassifierKind` /
  :data:`PrecisionMinerKind` enums,
* the runtime builds configs via :func:`build_classifier_config` /
  :func:`build_precision_miner_config`,
* the API publishes :func:`classifier_catalog` / :func:`miner_catalog` on
  ``/health`` so the UI can render dropdowns without hardcoding anything.

To add a new kind:

1. Add its config class (and output schema) under ``workflows/<family>/``.
2. Add one :class:`WorkflowKind` entry to :data:`CLASSIFIERS` or :data:`MINERS`.

That single entry updates the enums, builders, API schema, and UI dropdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from episcope.workflows.classification.config import (
    DataAccessibilityClassifierConfig,
    DataTypeClassifierConfig,
    GeoClassifierConfig,
    PaperTypeClassifierConfig,
)
from episcope.workflows.precision_miner.config import (
    FindDataSourcesConfig,
    FindSupplementaryLinksConfig,
    IdentifyKeyReferencesConfig,
)

if TYPE_CHECKING:
    from episcope.workflows.classification.config import BaseClassifierConfig
    from episcope.workflows.precision_miner.config import PrecisionMinerConfig


@dataclass(frozen=True)
class WorkflowKind:
    """One selectable workflow variant (a classifier or a miner).

    ``factory`` is a zero-argument callable (normally the config class) that
    returns a fresh config instance.
    """

    key: str
    label: str
    factory: Callable[[], Any]
    description: str = ""


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


# Generated enums: the single source consumed by the CLI (Typer) and API
# (Pydantic). Members and values are the registry keys.
ClassifierKind = Enum("ClassifierKind", {key: key for key in CLASSIFIERS}, type=str)
PrecisionMinerKind = Enum(
    "PrecisionMinerKind", {key: key for key in MINERS}, type=str
)


def _normalize(kind: object) -> str:
    return kind.value if isinstance(kind, Enum) else str(kind)


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
    """Serializable ``[{key, label, description}]`` for clients (e.g. the UI)."""
    return [
        {"key": spec.key, "label": spec.label, "description": spec.description}
        for spec in CLASSIFIERS.values()
    ]


def miner_catalog() -> List[Dict[str, str]]:
    """Serializable ``[{key, label, description}]`` for clients (e.g. the UI)."""
    return [
        {"key": spec.key, "label": spec.label, "description": spec.description}
        for spec in MINERS.values()
    ]
