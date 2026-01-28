"""
Defines Pydantic models and supporting enums for the paper-classification workflow.

Design goals
------------
1) Validation barrier for untrusted LLM JSON output (strict, reproducible).
2) Canonical internal representation of classifier outputs.

Notes
-----
- This code targets Pydantic v2.x.
- Letter codes (A, B, …) are kept for prompt compatibility, while enums provide
  stable semantic labels for downstream use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# Base output schema (shared by all classifiers)
# -----------------------------------------------------------------------------

class ClassificationOutput(BaseModel):
    """Base output schema returned by an LLM classifier."""
    reasoning: str = Field(..., description="Short 1–3 sentence explanation grounded in the provided text.")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence that the label set is correct.",
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Optional mapping from class code (e.g., 'A') to probability in [0,1]. "
            "If provided, values should sum to ~1."
        ),
    )


# -----------------------------------------------------------------------------
#  (Epidemiological parameter-estimation paper taxonomy)
# -----------------------------------------------------------------------------

class PaperType(str, Enum):
    """Parameter-estimation focused paper-type taxonomy.

    Notes
    -----
    - These labels capture *how the estimate is produced*: original data, inferred via models,
      synthesized across studies, or focused on method/tool development.
    - `UNCLEAR` is an internal fallback and should not be emitted by the LLM unless truly unavoidable.
    """
    PRIMARY_EMPIRICAL = "primary_empirical"
    MODELING_INFERENCE = "modeling_inference"
    META_ANALYSIS = "meta_analysis"
    METHODOLOGICAL = "methodological"
    REVIEW_COMMENTARY = "review_commentary"

    # Internal fallback (pipeline), not for LLM output
    UNCLEAR = "unclear"


PaperTypeCode = Literal["A", "B", "C", "D", "E", "F"]

_CODE_TO_ENUM: Dict[PaperTypeCode, PaperType] = {
    "A": PaperType.PRIMARY_EMPIRICAL,
    "B": PaperType.MODELING_INFERENCE,
    "C": PaperType.META_ANALYSIS,
    "D": PaperType.METHODOLOGICAL,
    "E": PaperType.REVIEW_COMMENTARY,
    "F": PaperType.UNCLEAR,
}

_CODE_LABELS: Dict[PaperTypeCode, str] = {
    "A": "Primary_Empirical",
    "B": "Modeling_Inference",
    "C": "Meta_Analysis",
    "D": "Methodological",
    "E": "Review_Commentary",
    "F": "Unclear / Not Specified",
}

_CODE_DEFINITIONS: Dict[PaperTypeCode, str] = {
    "A": "Uses original patient/field/surveillance data to estimate one or more epidemiological parameters.",
    "B": "Infers parameters from existing data using mathematical/statistical models (e.g., renewal, SEIR fitting, Bayesian inference).",
    "C": "Systematic review with quantitative pooling of parameter estimates across studies (meta-analysis).",
    "D": "Introduces, validates, or compares estimation methods and/or releases software/tools for parameter estimation.",
    "E": "Narrative review, commentary, or perspective discussing parameter values/interpretation without new data or pooling.",
    "F": "Cannot be confidently categorized from the available text.",
}


class PaperTypeClassificationOutput(ClassificationOutput):
    classification: PaperTypeCode = Field(
        ...,
        description="A single letter code (A–F) for the  taxonomy.",
    )


# -----------------------------------------------------------------------------
# DAVAIL (Data Availability) – protocol-aligned taxonomy
# -----------------------------------------------------------------------------

class DataAccessibility(str, Enum):
    """
    Protocol-aligned data availability labels (DAVAIL).

    IMPORTANT:
    - These labels are assigned strictly from what is stated in the paper.
    - There is no external verification of links or repositories.
    - `UNCLEAR` is reserved as an internal fallback and should not be emitted by the LLM.
    """
    OPEN = "open"
    AVAILABLE_UPON_REQUEST = "available_upon_request"
    REFERENCED = "referenced"
    CLOSED = "closed"
    NOT_STATED = "not_stated"

    # Internal fallback (pipeline), not for LLM output
    UNCLEAR = "unclear"


DataAccessibilityCode = Literal["A", "B", "C", "D", "E"]

DATA_ACCESS_CODE_TO_ENUM: Dict[DataAccessibilityCode, DataAccessibility] = {
    "A": DataAccessibility.OPEN,
    "B": DataAccessibility.AVAILABLE_UPON_REQUEST,
    "C": DataAccessibility.REFERENCED,
    "D": DataAccessibility.CLOSED,
    "E": DataAccessibility.NOT_STATED,
}

DATA_ACCESS_CODE_LABELS: Dict[DataAccessibilityCode, str] = {
    "A": "OPEN",
    "B": "AVAILABLE_UPON_REQUEST",
    "C": "REFERENCED",
    "D": "CLOSED",
    "E": "NOT_STATED",
}

DATA_ACCESS_CODE_DEFINITIONS: Dict[DataAccessibilityCode, str] = {
    "A": (
        "The paper claims data are available in a public repository with a stable link/identifier "
        "and/or provides reusable supplementary files (e.g., CSV tables) sufficient for reuse. This includes cases "
        "where the paper itself is the primary data release (e.g., descriptive studies) and a third party can "
        "reconstruct the raw dataset (e.g., line-list rows or time-series points) directly from the text/tables."
    ),
    "B": "The paper explicitly states data are available upon request, approval, or via an institutional process (contact author, data custodian, DUA).",
    "C": (
        "The paper attributes data to third-party sources (dashboards, public health authorities, institutional databases), "
        "but does NOT provide an access path or retrieval mechanism sufficient for a reader to obtain the data in the same manner. "
        "This also includes primary data collected by the authors where raw observations are not reported and no external access mechanism is provided."
    ),
    "D": "The paper explicitly states data cannot be shared or are restricted (legal/ethical/confidentiality/licensing), without an actionable access mechanism.",
    "E": "The paper does not provide enough information to determine data availability.",
}


class DataAccessibilityClassificationOutput(ClassificationOutput):
    """
    Multi-label output for DAVAIL.

    Use multiple labels when the paper uses multiple datasets with different
    availability mechanisms. If the paper is itself the primary data release,
    treat this case as OPEN if the raw dataset can be reconstructed from reported tables/appendices.
    """
    classification: List[DataAccessibilityCode] = Field(
        ...,
        min_length=1,
        description=(
            "List of one or more letters among: A (OPEN), B (AVAILABLE_UPON_REQUEST), C (REFERENCED), D (CLOSED), E (NOT_STATED)."
        ),
    )
    primary_label: Optional[DataAccessibilityCode] = Field(
        default=None,
        description="Optional dominant label if one clearly dominates; otherwise null.",
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "DataAccessibilityClassificationOutput":
        # Deduplicate while preserving first-seen order, then sort into canonical order A–F.
        seen = []
        for c in self.classification:
            if c not in seen:
                seen.append(c)

        canonical = [c for c in ["A", "B", "C", "D", "E"] if c in seen]

        # If primary_label is present but not in classification, drop it.
        if self.primary_label is not None and self.primary_label not in canonical:
            self.primary_label = None

        # If only one label remains, set primary_label if missing.
        if len(canonical) == 1 and self.primary_label is None:
            self.primary_label = canonical[0]

        self.classification = canonical
        return self


# -----------------------------------------------------------------------------
# GEO (Geography) – protocol-aligned taxonomy (continent-level + IRRELEVANT/UNCLEAR)
# -----------------------------------------------------------------------------

class GeoRegion(str, Enum):
    AFRICA = "africa"
    ASIA = "asia"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA = "oceania"

    IRRELEVANT = "irrelevant"  # e.g., strictly controlled laboratory setting, or purely synthetic/no-geo data
    UNCLEAR = "unclear"        # not enough stated information


GeoCode = Literal["A", "B", "C", "D", "E", "F", "G", "H"]

GEO_CODE_TO_ENUM: Dict[GeoCode, GeoRegion] = {
    "A": GeoRegion.AFRICA,
    "B": GeoRegion.ASIA,
    "C": GeoRegion.EUROPE,
    "D": GeoRegion.NORTH_AMERICA,
    "E": GeoRegion.SOUTH_AMERICA,
    "F": GeoRegion.OCEANIA,
    "G": GeoRegion.IRRELEVANT,
    "H": GeoRegion.UNCLEAR,
}

GEO_CODE_LABELS: Dict[GeoCode, str] = {
    "A": "Africa",
    "B": "Asia",
    "C": "Europe",
    "D": "North America",
    "E": "South America",
    "F": "Oceania",
    "G": "IRRELEVANT",
    "H": "UNCLEAR / Not specified",
}

GEO_CODE_DEFINITIONS: Dict[GeoCode, str] = {
    "A": "Africa",
    "B": "Asia",
    "C": "Europe",
    "D": "North America",
    "E": "South America",
    "F": "Oceania",
    "G": "IRRELEVANT (e.g., strictly controlled lab setting, or no meaningful geographic origin stated)",
    "H": "UNCLEAR / Not specified",
}


class GeoExtras(BaseModel):
    countries: List[str] = Field(
        default_factory=list,
        description="Country names explicitly tied to the epidemiological evidence/data used in the paper."
    )
    cities: List[str] = Field(
        default_factory=list,
        description="City/region/state/province names explicitly tied to the epidemiological evidence/data used in the paper."
    )

    @model_validator(mode="after")
    def _normalize(self) -> "GeoExtras":
        def _clean(xs: Sequence[str]) -> List[str]:
            out: List[str] = []
            for x in xs:
                if not isinstance(x, str):
                    continue
                s = " ".join(x.split()).strip()
                if not s:
                    continue
                if s not in out:
                    out.append(s)
            return out

        self.countries = _clean(self.countries)
        self.cities = _clean(self.cities)
        return self


class GeoClassificationOutput(ClassificationOutput):
    classification: List[GeoCode] = Field(
        ...,
        min_length=1,
        description=(
            "List of letters among: A Africa, B Asia, C Europe, D North America, "
            "E South America, F Oceania, G IRRELEVANT, H UNCLEAR."
        ),
    )
    primary_label: Optional[GeoCode] = Field(
        default=None,
        description="Dominant continent if one clearly dominates; otherwise null."
    )
    extras: GeoExtras = Field(default_factory=GeoExtras, description="Extracted geographic locations.")

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "GeoClassificationOutput":
        # Deduplicate, then sort into canonical order.
        seen = []
        for c in self.classification:
            if c not in seen:
                seen.append(c)
        canonical = [c for c in ["A", "B", "C", "D", "E", "F", "G", "H"] if c in seen]

        # IRRELEVANT dominates; UNCLEAR is only used when no other labels remain.
        if "G" in canonical:
            canonical = ["G"]
        elif "H" in canonical and len(canonical) > 1:
            canonical = [c for c in canonical if c != "H"]

        if not canonical:
            canonical = ["H"]

        if self.primary_label is not None and self.primary_label not in canonical:
            self.primary_label = None

        if len(canonical) == 1 and self.primary_label is None:
            self.primary_label = canonical[0]

        self.classification = canonical
        return self


# -----------------------------------------------------------------------------
# DTYPE (Data Type) – protocol-aligned taxonomy (#Data4COVID19 families + synthetic/no data)
# -----------------------------------------------------------------------------

class DataType(str, Enum):
    """Protocol-aligned data type categories (DTYPE)."""
    TRADITIONAL = "traditional"
    NON_TRADITIONAL_HEALTH = "non_traditional_health"
    NON_TRADITIONAL_MOBILITY = "non_traditional_mobility"
    NON_TRADITIONAL_SENTIMENT = "non_traditional_sentiment"
    NON_TRADITIONAL_ECONOMIC = "non_traditional_economic"
    SYNTHETIC = "synthetic"
    NO_EMPIRICAL_DATA = "no_empirical_data"

    # Internal fallback
    UNCLEAR = "unclear"


DataTypeCode = Literal["A", "B", "C", "D", "E", "F", "G", "H"]

DATA_TYPE_CODE_TO_ENUM: Dict[DataTypeCode, DataType] = {
    "A": DataType.TRADITIONAL,
    "B": DataType.NON_TRADITIONAL_HEALTH,
    "C": DataType.NON_TRADITIONAL_MOBILITY,
    "D": DataType.NON_TRADITIONAL_SENTIMENT,
    "E": DataType.NON_TRADITIONAL_ECONOMIC,
    "F": DataType.SYNTHETIC,
    "G": DataType.NO_EMPIRICAL_DATA,
    "H": DataType.UNCLEAR,
}

DATA_TYPE_CODE_LABELS: Dict[DataTypeCode, str] = {
    "A": "Traditional",
    "B": "Non-Traditional Health",
    "C": "Non-Traditional Mobility",
    "D": "Non-Traditional Sentiment",
    "E": "Non-Traditional Economic",
    "F": "Synthetic",
    "G": "No Empirical Data",
    "H": "Unclear / Not specified",
}

DATA_TYPE_CODE_DEFINITIONS: Dict[DataTypeCode, str] = {
    "A": "Traditional – established epidemiological/public-health surveillance & clinical/administrative data used in analysis.",
    "B": "Non-Traditional Health – novel/non-standard health proxies or platforms (symptom apps, wearables, wastewater, large-scale digital patient platforms).",
    "C": "Non-Traditional Mobility – mobility/proximity signals (CDRs, GPS/SDK mobility, Bluetooth proximity) used as epidemiological proxies.",
    "D": "Non-Traditional Sentiment – social listening/surveys/crowdsourced perception & behavior proxies used as data.",
    "E": "Non-Traditional Economic – economic/financial data (transaction data, supply-chain/shipment data).",
    "F": "Synthetic – simulated data used as evidence supporting claims (e.g., scenario projections, counterfactuals).",
    "G": "No Empirical Data – conceptual/theoretical/methodological work with no empirical dataset analyzed.",
    "H": "Unclear / Not specified",

    
}


class DataTypeClassificationOutput(ClassificationOutput):
    """Multi-label output for DTYPE."""
    classification: List[DataTypeCode] = Field(
        ...,
        min_length=1,
        description=(
            "List of one or more labels among: A–H. "
            "Use multiple labels if multiple data types are used. "
            "G (No Empirical Data) is mutually exclusive with all others."
        ),
    )
    primary_label: Optional[DataTypeCode] = Field(
        default=None,
        description="Optional dominant label if one clearly dominates; otherwise null."
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "DataTypeClassificationOutput":
        # Deduplicate then canonical order.
        seen = []
        for c in self.classification:
            if c not in seen:
                seen.append(c)
        canonical = [c for c in ["A", "B", "C", "D", "E", "F", "G", "H"] if c in seen]

        # NO_EMPIRICAL_DATA dominates.
        if "G" in canonical:
            canonical = ["G"]
        # UNCLEAR only when no other label remains.
        elif "H" in canonical and len(canonical) > 1:
            canonical = [c for c in canonical if c != "H"]

        if not canonical:
            canonical = ["H"]

        if self.primary_label is not None and self.primary_label not in canonical:
            self.primary_label = None

        if len(canonical) == 1 and self.primary_label is None:
            self.primary_label = canonical[0]

        self.classification = canonical
        return self


# -----------------------------------------------------------------------------
# Final workflow internal result object (used outside Pydantic boundary)
# -----------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    classification: List[Any]
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
