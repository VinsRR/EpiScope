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

# class PaperType(str, Enum):
#     """Parameter-estimation focused paper-type taxonomy.

#     Notes
#     -----
#     - These labels capture *how the estimate is produced*: original data, inferred via models,
#       synthesized across studies, or focused on method/tool development.
#     - `UNCLEAR` is an internal fallback and should not be emitted by the LLM unless truly unavoidable.
#     """
#     PRIMARY_EMPIRICAL = "primary_empirical"
#     MODELING_INFERENCE = "modeling_inference"
#     META_ANALYSIS = "meta_analysis"
#     METHODOLOGICAL = "methodological"
#     REVIEW_COMMENTARY = "review_commentary"

#     # Internal fallback (pipeline), not for LLM output
#     UNCLEAR = "unclear"


# PaperTypeCode = Literal["A", "B", "C", "D", "E", "F"]

# _CODE_TO_ENUM: Dict[PaperTypeCode, PaperType] = {
#     "A": PaperType.PRIMARY_EMPIRICAL,
#     "B": PaperType.MODELING_INFERENCE,
#     "C": PaperType.META_ANALYSIS,
#     "D": PaperType.METHODOLOGICAL,
#     "E": PaperType.REVIEW_COMMENTARY,
#     "F": PaperType.UNCLEAR,
# }

# _CODE_LABELS: Dict[PaperTypeCode, str] = {
#     "A": "Primary_Empirical",
#     "B": "Modeling_Inference",
#     "C": "Meta_Analysis",
#     "D": "Methodological",
#     "E": "Review_Commentary",
#     "F": "Unclear / Not Specified",
# }

# _CODE_DEFINITIONS: Dict[PaperTypeCode, str] = {
#     "A": "Uses original patient/field/surveillance data to estimate one or more epidemiological parameters.",
#     "B": "Infers parameters from existing data using mathematical/statistical models (e.g., renewal, SEIR fitting, Bayesian inference).",
#     "C": "Systematic review with quantitative pooling of parameter estimates across studies (meta-analysis).",
#     "D": "Introduces, validates, or compares estimation methods and/or releases software/tools for parameter estimation.",
#     "E": "Narrative review, commentary, or perspective discussing parameter values/interpretation without new data or pooling.",
#     "F": "Cannot be confidently categorized from the available text.",
# }


# class PaperTypeClassificationOutput(ClassificationOutput):
#     classification: PaperTypeCode = Field(
#         ...,
#         description="A single letter code (A–F) for the  taxonomy.",
#     )


# -----------------------------------------------------------------------------
# PAPER_TYPE (Epidemiological parameter-estimation paper taxonomy)
# -----------------------------------------------------------------------------

class PaperType(str, Enum):
    """Epidemiological parameter-estimation paper taxonomy (PAPER_TYPE).

    PAPER_TYPE classifies each document by the kind of scientific contribution it makes
    to epidemiological parameter estimation.

    Notes
    -----
    - Assigned at the *paper level* (not dataset level).
    - Primary + secondary: assign exactly one primary_label and optional secondary_labels using this label set.
    - `UNCLEAR` is an internal fallback and should not be emitted by the LLM unless truly unavoidable.
    """

    EMPIRICAL = "EMPIRICAL"
    INFERENCE = "INFERENCE"
    FORECAST = "FORECAST"
    SYNTHESIS = "SYNTHESIS"
    METHODOLOGICAL = "METHODOLOGICAL"
    COMMENTARY = "COMMENTARY"

    # Use when none of the above apply
    OTHER = "OTHER"

    # Internal fallback (pipeline), not for LLM output
    UNCLEAR = "UNCLEAR"


PAPER_TYPE_DEFINITIONS: Dict[str, str] = {
    "EMPIRICAL": (
        "Primary contribution is based on **original** empirical data collection, where claims rely on "
        "direct measurement from a clearly defined study population or sample (e.g., cohort/case–control/cross-sectional "
        "studies, randomized trials, surveillance/outbreak reports, or data papers describing new datasets)."
    ),
    "INFERENCE": (
        "Primary contribution is estimating epidemiological parameters or reconstructing unobserved epidemic processes "
        "by fitting explicit statistical or mechanistic models to observed data; results depend on model structure and assumptions "
        "(e.g., estimating R0 or generation time from case data, reconstructing transmission chains, phylodynamic inference, "
        "or estimating seroconversion rates from serological surveys)."
    ),
    "FORECAST": (
        "Primary contribution is prediction or scenario analysis: forward projections of epidemic trajectories, comparisons of "
        "intervention scenarios, or counterfactual simulations; outputs are projected future outcomes or policy comparisons under stated assumptions "
        "(e.g., policy-planning projections, intervention scenario comparisons, or real-time forecasting submissions)."
    ),
    "SYNTHESIS": (
        "Primary contribution is a systematic synthesis of existing evidence using predefined search, selection, and appraisal criteria "
        "(e.g., quantitative meta-analyses pooling estimates across studies or PRISMA-compliant systematic reviews)."
    ),
    "METHODOLOGICAL": (
        "Primary contribution is development, evaluation, or comparison of methods (e.g., new estimators, inference frameworks, "
        "identifiability/bias analyses, software tools, or theoretical results about epidemic processes such as epidemic thresholds)."
    ),
    "COMMENTARY": (
        "Primary contribution is discussion or interpretation without presenting new empirical estimates, model-based inference, or systematic evidence synthesis "
        "(e.g., narrative literature reviews, perspective pieces, editorials, or policy commentaries)."
    ),
    "OTHER": "Does not fit any of the categories above.",
    "UNCLEAR": "Internal fallback when the provided excerpt is insufficient to classify.",
}



class PaperTypeClassificationOutput(ClassificationOutput):
    """Primary + secondary output for PAPER_TYPE.

    Papers often combine multiple tasks (data collection, parameter inference, forecasting).
    This schema distinguishes the paper's *primary contribution* (main deliverable) from any
    *secondary contributions* (supporting tasks that enable or accompany the primary).

    Use OTHER when none of the substantive categories apply.
    UNCLEAR is an internal fallback and should be avoided unless the excerpt is genuinely insufficient.
    """

    primary_label: PaperType = Field(
        ...,
        description=(
            "Exactly one PAPER_TYPE label capturing the paper's primary scientific contribution."
        ),
    )
    secondary_labels: List[PaperType] = Field(
        default_factory=list,
        description=(
            "Optional secondary PAPER_TYPE labels for additional substantial contributions. "
            "Do not include the primary_label here. Avoid OTHER/UNCLEAR unless necessary."
        ),
    )

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "PaperTypeClassificationOutput":
        # Normalize secondary labels: deduplicate (preserve order), drop primary/UNCLEAR, and enforce OTHER exclusivity.
        if self.primary_label in {PaperType.UNCLEAR, PaperType.OTHER}:
            self.secondary_labels = []
            return self

        seen: List[PaperType] = []
        for lbl in self.secondary_labels or []:
            if lbl not in seen:
                seen.append(lbl)

        # Drop primary and UNCLEAR from secondary list
        canonical = [lbl for lbl in seen if lbl not in {self.primary_label, PaperType.UNCLEAR}]

        # OTHER should be exclusive: if any substantive label exists, drop OTHER from secondary
        canonical = [lbl for lbl in canonical if lbl != PaperType.OTHER]

        self.secondary_labels = canonical
        return self





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
    REPORTED = "reported"
    AVAILABLE_UPON_REQUEST = "available_upon_request"
    REFERENCED = "referenced"
    CLOSED = "closed"
    NOT_STATED = "not_stated"

    # Internal fallback (pipeline), not for LLM output
    UNCLEAR = "unclear"


DataAccessibilityCode = Literal["A", "B", "C", "D", "E", "F"]

DATA_ACCESS_CODE_TO_ENUM: Dict[DataAccessibilityCode, DataAccessibility] = {
    "A": DataAccessibility.OPEN,
    "B": DataAccessibility.REPORTED,
    "C": DataAccessibility.REFERENCED,
    "D": DataAccessibility.AVAILABLE_UPON_REQUEST,
    "E": DataAccessibility.CLOSED,
    "F": DataAccessibility.NOT_STATED,
}

DATA_ACCESS_CODE_LABELS: Dict[DataAccessibilityCode, str] = {
    "A": "OPEN",
    "B": "REPORTED",
    "C": "REFERENCED",
    "D": "AVAILABLE_UPON_REQUEST",
    "E": "CLOSED",
    "F": "NOT_STATED",
}

DATA_ACCESS_CODE_DEFINITIONS: Dict[DataAccessibilityCode, str] = {
    "A": (
        "The data underlying the results are publicly available via a persistent identifier (e.g., DOI, accession number), "
        "a stable repository link, and/or by being fully released within the paper package (e.g., full line-lists or complete contingency tables) "
        "in a form sufficient for a third party to reconstruct the dataset for new analyses. Mentions of supplementary material only qualify if they "
        "contain the specific, extractable data used for the work (the link/identifier must point to the underlying data supporting the results, not merely "
        "state that supplementary information exists)."
    ),
    "B": (
        "The dataset is not fully released, but the paper provides granular, extractable data values that allow a third party to independently re-aggregate "
        "and recompute at least one key quantity (e.g., a rate or proportion) from the reported values alone, without relying on narrative summaries or derived "
        "model outputs. This label depends on the richness of the in-paper reporting, regardless of whether the source data are internal or third-party."
    ),
    "C": (
        "The data are attributed to a specifically identifiable third-party source (e.g., public health authority, institutional database, emergency dashboard, "
        "named public source, or other paper), but the paper provides neither a stable retrieval path for the source data nor sufficient detail to re-aggregate "
        "the data (i.e., it fails the REPORTED re-aggregation criterion). Reporting is data-thin—typically limited to derived outputs such as effect sizes, "
        "coefficients, or fitted model curves—preventing independent verification of the underlying counts. Vague attributions (e.g., \"government data\") "
        "do not qualify."
    ),
    "D": (
        "The paper explicitly provides a request-based mechanism to access the data (e.g., contact the author, apply to a data access committee, or follow a "
        "specific institutional process)."
    ),
    "E": (
        "The paper explicitly states that the data cannot be shared or are restricted due to legal, ethical, confidentiality, or licensing constraints, without "
        "providing an actionable access mechanism."
    ),
    "F": (
        "The paper provides no data access statement, no repository link, no request mechanism, and no identifiable third-party attribution, while also failing "
        "to provide extractable values sufficient for REPORTED. Statements that data were collected/generated, without an access mechanism and without "
        "re-aggregatable values, should be labeled NOT_STATED."
    ),
}


class DataAccessibilityClassificationOutput(ClassificationOutput):
    """
    Multi-label output for Data Availability

    Use multiple labels when the paper uses multiple datasets with different
    availability mechanisms. If the paper is itself the primary data release,
    treat this case as OPEN if the raw dataset can be reconstructed from reported tables/appendices.
    """
    classification: List[DataAccessibilityCode] = Field(
        ...,
        min_length=1,
        description=(
            "List of one or more letters among: A (OPEN), B (REPORTED), C (REFERENCED), D (AVAILABLE_UPON_REQUEST), E (CLOSED), F (NOT_STATED)."
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

        canonical = [c for c in ["A", "B", "C", "D", "E", "F"] if c in seen]

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
