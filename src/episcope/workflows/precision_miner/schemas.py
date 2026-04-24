"""
Defines the Pydantic models for the precision miner workflow.

These models serve a dual purpose:
1.  **Validation:** They act as a protective barrier, parsing and validating the
    raw, untrusted JSON output from an external source like an LLM.
2.  **Internal Data Structure:** They are the single, canonical source of truth
    for the workflow's output. Once validated, these models are used directly
    by the rest of the application.

This unified approach avoids the need to maintain separate dataclasses and
Pydantic models, simplifying the codebase.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# --- Main Workflow Output Models ---


class ExtractionItem(BaseModel):
    name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    raw_text: Optional[str] = None


class ExtractionResult(BaseModel):
    description: str
    items: List[ExtractionItem] = Field(default_factory=list)


class DataSource(BaseModel):
    source_name: str
    url: str = "N/A"
    explanation: str = ""
    section_found: str = ""


# --- Pydantic Schemas for LLM Validation ---


class ExtractionItemSchema(BaseModel):
    name: str = Field(..., description="Full source name as appears in paper")
    url: Optional[str] = Field(None, description="URL/DOI if available else null")
    explanation: str = Field(..., description="Why this item is key")
    raw_text: Optional[str] = Field(
        None, description="Exact reference snippet from the paper"
    )


class ExtractionResultSchema(BaseModel):
    description: str = Field(
        ..., description="Short summary of why these references were selected"
    )
    items: List[ExtractionItemSchema] = Field(
        default_factory=list, description="List of key references"
    )
