from enum import Enum
from typing import Dict, Optional, Literal

from pydantic import BaseModel, Field

class PaperType(Enum):
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis"
    UNCLEAR = "unclear"
    # METHODS_TOOLS = "methods_tools"
    # CASE_STUDY = "case_study"
    # COMMENTARY = "commentary"
    # OTHER = "other"
    # UNCLEAR = "unclear"

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



# Pydantic schema for structured LLM output
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