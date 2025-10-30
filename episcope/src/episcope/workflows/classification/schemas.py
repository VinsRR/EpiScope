"""
Defines the Pydantic models for validating the external, untrusted output
of the classification workflow, typically from a Large Language Model (LLM).

These models act as a protective barrier, ensuring that the raw JSON output
from the LLM conforms to a strict, expected structure before it is processed
further. They are not meant to be the internal domain models of the application.

Once an LLM's output is successfully parsed and validated by these models, the
workflow's logic then maps this validated data into the clean, internal
dataclasses defined in `results.py`. This separation ensures that the rest of
the application only ever interacts with reliable, well-defined data structures.
"""
from typing import Dict, Optional, Literal

from pydantic import BaseModel, Field

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