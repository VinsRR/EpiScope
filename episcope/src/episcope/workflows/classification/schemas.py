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
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# --- Controlled Vocabulary ---

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

# --- Pydantic Schemas for LLM Validation ---

class ClassificationOutput(BaseModel):
    classification: str = Field(..., description="A single letter representing the classification")
    reasoning: str = Field(..., description="Short 1-2 sentence explanation")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    class_probabilities: Optional[Dict[str, float]] = None

class PaperTypeClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B","C"] = Field(..., description="One of: A, B, C")

class DataAccessibilityClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B", "C"] = Field(..., description="One of: A, B, C")

class DataNationClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., description="One of: A, B, C, D, E, F, G")

class DataTypeClassificationOutput(ClassificationOutput):
    classification: Literal["A", "B", "C", "D"] = Field(..., description="One of: A, B, C, D")

# --- Final Workflow Output Model ---

@dataclass
class ClassificationResult:
    classification: Any
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)
