"""
Defines the internal, clean data models for the precision miner workflow.

These dataclasses represent the "source of truth" for the workflow's output.
They are what the rest of the application should expect to receive and work with
after an extraction has been performed.

This is distinct from the Pydantic models in `schemas.py`, which are used only
for parsing and validating the raw, untrusted output from an external source
like an LLM before these clean internal models are created.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# --- Result Dataclasses ---

@dataclass
class DataSource:
    source_name: str
    url: str = "N/A"
    explanation: str = ""
    section_found: str = ""

@dataclass
class ExtractionItem:
    name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    raw_text: Optional[str] = None

@dataclass
class ExtractionResult():
    description: str
    items: List[ExtractionItem] = field(default_factory=list)