"""
core/provenance.py

Data models capturing provenance information for generated answers.

Every answer returned by EpiScope includes structured metadata
identifying the source of each supporting fact.  Provenance metadata
includes the paper identifier, a snippet of text, section locator,
index version, model identifier and prompt identifier.  These
structures are used internally and also surfaced via the API and UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Evidence:
    paper_id: str
    snippet: str
    section: Optional[str] = None
    index_version: Optional[str] = None
    model_id: Optional[str] = None
    prompt_id: Optional[str] = None


@dataclass
class Provenance:
    answer: str
    evidences: List[Evidence] = field(default_factory=list)

