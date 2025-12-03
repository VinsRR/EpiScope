"""
Defines the Pydantic models for the classification workflow.

These models serve a dual purpose:
1.  **Validation:** They act as a protective barrier, parsing and validating the
    raw, untrusted JSON output from an external source like an LLM.
2.  **Internal Data Structure:** They are the single, canonical source of truth
    for the workflow's output. Once validated, these models are used directly
    by the rest of the application.

This unified approach avoids the need to maintain separate dataclasses and
Pydantic models, simplifying the codebase.
"""
from enum import Enum
from typing import Any, Dict, Optional, Literal, List
from pydantic import BaseModel, Field
from dataclasses import dataclass, field



# --- Pydantic Schemas for LLM Validation ---

class ClassificationOutput(BaseModel):
    """Base output schema (kept if you reuse this for other classifiers)."""
    reasoning: str = Field(..., description="Short 1–3 sentence explanation")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall confidence that the label set is correct."
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Optional mapping from class code to probability in [0,1]. Since these are probabilities, their sum should be 1."
    )


##########################################################################################
############################## PAPER TYPE ################################################
##########################################################################################

class PaperType(Enum):
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis"
    UNCLEAR = "unclear"


class PaperTypeClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B","C"] = Field(..., description="A single letter classification, one of: A, B, C")


##########################################################################################
############################# DATA ACCESSIBILITY #########################################
##########################################################################################



# class DataAccessibility(str, Enum):
#     OPEN_ACCESS = "open_access"
#     RESTRICTED_ACCESS = "restricted_access"
#     NOT_AVAILABLE = "not_available"
#     UNCLEAR = "unclear"


# DATA_ACCESS_CODE = Literal["A", "B", "C", "D"]

# DATA_ACCESS_CODE_TO_ENUM: Dict[DATA_ACCESS_CODE, DataAccessibility] = {
#     "A": DataAccessibility.OPEN_ACCESS,
#     "B": DataAccessibility.RESTRICTED_ACCESS,
#     "C": DataAccessibility.NOT_AVAILABLE,
#     "D": DataAccessibility.UNCLEAR,
# }

# DATA_ACCESS_CODE_LABELS = {
#     "A": "Open Access (public repository, no permission required)",
#     "B": "Restricted Access (DUA / approval / request required)",
#     "C": "Not Available (explicitly not sharable, proprietary, OR no dataset used)",
#     "D": "Unclear (insufficient evidence about data sharing)",
# }


# class DataAccessibilityClassificationOutput(ClassificationOutput):
#     classification: DATA_ACCESS_CODE = Field(
#         ...,
#         description="A single letter: A (Open), B (Restricted), C (Not Available), D (Unclear)"
#     )


from enum import Enum
from typing import Dict, Literal
from pydantic import BaseModel, Field


class DataAccessibility(str, Enum):
    OPEN_ACCESS = "open_access"       # public repo/URL, no permission
    UPON_REQUEST = "upon_request"     # available, but only via request/DUA/approval
    NOT_AVAILABLE = "not_available"   # cannot be shared or no dataset exists
    NOT_STATED = "not_stated"         # paper does not say how/if data are shared
    UNCLEAR = "unclear"               # internal fallback (pipeline), not for LLM


DataAccessibilityCode = Literal["A", "B", "C", "D"]

DATA_ACCESS_CODE_TO_ENUM: Dict[DataAccessibilityCode, DataAccessibility] = {
    "A": DataAccessibility.OPEN_ACCESS,
    "B": DataAccessibility.UPON_REQUEST,
    "C": DataAccessibility.NOT_AVAILABLE,
    "D": DataAccessibility.NOT_STATED,
}

DATA_ACCESS_CODE_LABELS: Dict[DataAccessibilityCode, str] = {
    "A": "OPEN_ACCESS – data are in a public repository/URL, no permission required.",
    "B": "UPON_REQUEST – data are available but only via request, approval, or data use agreement.",
    "C": "NOT_AVAILABLE – data cannot be shared (proprietary, legal/ethical limits) or no dataset exists.",
    "D": "NOT_STATED – the text does not explain whether/how the data are available.",
}


class DataAccessibilityClassificationOutput(ClassificationOutput):
    classification: DataAccessibilityCode = Field(
        ...,
        description="A single letter: A (OPEN_ACCESS), B (UPON_REQUEST), "
                    "C (NOT_AVAILABLE), or D (NOT_STATED).",
    )

##########################################################################################
################################## GEO PROVENANCE ########################################
##########################################################################################



# class GeoRegion(str, Enum):
#     AFRICA = "africa"
#     ASIA = "asia"
#     EUROPE = "europe"
#     NORTH_AMERICA = "north_america"
#     SOUTH_AMERICA = "south_america"
#     OCEANIA = "oceania"
#     SYNTHETIC = "synthetic"
#     UNCLEAR = "unclear"


# GEO_CODE = Literal[
#     "A", "B", "C", "D", "E", "F", "G", "H"
# ]

# GEO_CODE_TO_ENUM = {
#     "A": GeoRegion.AFRICA,
#     "B": GeoRegion.ASIA,
#     "C": GeoRegion.EUROPE,
#     "D": GeoRegion.NORTH_AMERICA,
#     "E": GeoRegion.SOUTH_AMERICA,
#     "F": GeoRegion.OCEANIA,
#     "G": GeoRegion.SYNTHETIC,
#     "H": GeoRegion.UNCLEAR,
# }

# GEO_CODE_LABELS = {
#     "A": "Africa",
#     "B": "Asia",
#     "C": "Europe",
#     "D": "North America",
#     "E": "South America",
#     "F": "Oceania",
#     "G": "Synthetic",
#     "H": "Unclear / Not specified",
# }

class GeoRegion(str, Enum):
    AFRICA = "africa"
    ASIA = "asia"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA = "oceania"
    SYNTHETIC = "synthetic"   # data are purely synthetic, no real geo origin
    UNCLEAR = "unclear"       # not enough info


GeoCode = Literal["A", "B", "C", "D", "E", "F", "G", "H"]

GEO_CODE_TO_ENUM: Dict[GeoCode, GeoRegion] = {
    "A": GeoRegion.AFRICA,
    "B": GeoRegion.ASIA,
    "C": GeoRegion.EUROPE,
    "D": GeoRegion.NORTH_AMERICA,
    "E": GeoRegion.SOUTH_AMERICA,
    "F": GeoRegion.OCEANIA,
    "G": GeoRegion.SYNTHETIC,
    "H": GeoRegion.UNCLEAR,
}

GEO_CODE_LABELS: Dict[GeoCode, str] = {
    "A": "Africa",
    "B": "Asia",
    "C": "Europe",
    "D": "North America",
    "E": "South America",
    "F": "Oceania",
    "G": "Synthetic (no real geographic origin)",
    "H": "Unclear / Not specified",
}



# class GeoClassificationOutput(ClassificationOutput):
#     classification: GEO_CODE = Field(
#         ...,
#         description=(
#             "A single letter: "
#             "A Africa, B Asia, C Europe, D North America, E South America, "
#             "F Oceania, G Synthetic, H Unclear"
#         )
#     )
#     locations: Dict[str, List[str]] = Field(
#         default_factory=lambda: {"countries": [], "cities": []},
#         description="Extracted country and city/region names."
#     )


class GeoExtras(BaseModel):
    countries: List[str] = Field(default_factory=list, description = "A flat list of country names that are sources of the analyzed data.")
    cities: List[str] = Field(default_factory=list, description = "A flat list of cities or region names that are sources of the analyzed data.")


class GeoClassificationOutput(ClassificationOutput):
    classification: List[GeoCode] = Field(
        ...,
        min_items=1,
        description=(
            "List of letters among: "
            "A Africa, B Asia, C Europe, D North America, E South America, "
            "F Oceania, G Synthetic, H Unclear."
        ),
    )

    primary_label: GeoCode | None = Field(
        None,
        description="Dominant region if one clearly dominates; otherwise null."
    )

    extras: GeoExtras = Field(..., description="Extracted geographic locations (countries and cities/regions).")




##########################################################################################
##########################################################################################
##########################################################################################




class DataType(str, Enum):
    """Granular data type categories for epidemiological studies."""
    # Traditional data
    TRADITIONAL = "traditional"  # e.g. surveys, EHR, routine surveillance

    # Non-traditional data (taxonomy adapted from #Data4COVID19 review)
    NON_TRADITIONAL_HEALTH = "non_traditional_health"      # e.g. symptom apps, wearables, wastewater
    NON_TRADITIONAL_MOBILITY = "non_traditional_mobility"  # e.g. telecoms CDRs, GPS traces, app-based mobility
    NON_TRADITIONAL_ECONOMIC = "non_traditional_economic"  # e.g. card transactions, supply chain data
    NON_TRADITIONAL_SENTIMENT = "non_traditional_sentiment"  # e.g. social media, crowdsourced opinions

    # Other categories
    SYNTHETIC = "synthetic"  # fully simulated or synthetic datasets
    NO_EMPIRICAL_DATA = "no_empirical_data"  # conceptual/commentary; no data actually analyzed
    UNCLEAR = "unclear"  # not enough information to classify


# Letter codes (kept for backward-compatibility with A/B/… style labels)
DataTypeCode = Literal["A", "B", "C", "D", "E", "F", "G", "H"]

DATA_TYPE_CODE_TO_ENUM: Dict[DataTypeCode, DataType] = {
    "A": DataType.TRADITIONAL,
    "B": DataType.NON_TRADITIONAL_HEALTH,
    "C": DataType.NON_TRADITIONAL_MOBILITY,
    "D": DataType.NON_TRADITIONAL_ECONOMIC,
    "E": DataType.NON_TRADITIONAL_SENTIMENT,
    "F": DataType.SYNTHETIC,
    "G": DataType.NO_EMPIRICAL_DATA,
    "H": DataType.UNCLEAR,
}

DATA_TYPE_CODE_LABELS: Dict[DataTypeCode, str] = {
    "A": "Traditional (e.g., epidemiological surveys, EHR, registries)",
    "B": "Non-Traditional Health Data (e.g., symptom apps, wearables, wastewater, digital patient data)",
    "C": "Non-Traditional Mobility Data (e.g., telecom CDRs, GPS, SDK-derived mobility, Bluetooth proximity)",
    "D": "Non-Traditional Economic Data (e.g., card transactions, supply-chain, open contracting)",
    "E": "Non-Traditional Sentiment Data (e.g., social media, crowdsourced attitudes/opinions)",
    "F": "Synthetic Data (e.g., simulated populations, synthetic trajectories, agent-based simulation outputs)",
    "G": "No Empirical Data (e.g., conceptual, methodological, or narrative work with no analyzed dataset)",
    "H": "Unclear / Not Specified",
}



class DataTypeClassificationOutput(ClassificationOutput):
    """Multi-label output: one or more letter codes."""
    classification: List[DataTypeCode] = Field(
        ...,
        description=(
            "List of one or more labels among: A, B, C, D, E, F, G, H. "
            "Use multiple labels if the paper uses multiple data types."
        ),
        min_items=1,
    )
    primary_label: Optional[DataTypeCode] = Field(
        None,
        description=(
            "Optional single best label (A–H), if one type clearly dominates; "
            "otherwise leave null."
        ),
    )









class DataNationClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., description="A single letter classification, one of: A, B, C, D, E, F, G")


# --- Final Workflow Output Model ---

@dataclass
class ClassificationResult:
    # classification: Any
    classification: List[Any]  
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
