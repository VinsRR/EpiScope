"""
Defines the internal, clean data models for the classification workflow.

These dataclasses and enums represent the "source of truth" for the workflow's
output and its controlled vocabulary. They are what the rest of the application
should expect to receive and work with.

This is distinct from the Pydantic models in `schemas.py`, which are used only
for parsing and validating the raw, untrusted output from an external source
like an LLM before these clean internal models are created.
"""
from enum import Enum
from typing import Any, Dict
from dataclasses import dataclass, field

# --- Enums ---

class PaperType(Enum):
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis"
    UNCLEAR = "unclear"

class DataAccessibility(Enum):
    OPEN_ACCESS = "open_access"
    RESTRICTED_ACCESS = "restricted_access"
    NOT_AVAILABLE = "not_available"

class DataNation(Enum):
    USA = "usa"
    UK = "uk"
    CHINA = "china"
    EUROPE_MULTIPLE = "europe_multiple"
    GLOBAL = "global"
    SYNTHETIC = "synthetic"
    NOT_SPECIFIED = "not_specified"

class DataType(Enum):
    TRADITIONAL = "traditional"
    NON_TRADITIONAL = "non_traditional"
    SYNTHETIC = "synthetic"
    NOT_SPECIFIED = "not_specified"

# --- Result Dataclass ---

@dataclass
class ClassificationResult:
    classification: Any
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)