from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Provenance / evidence
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Full lineage record for one retrieved snippet."""

    paper_id: str
    snippet: str
    section: Optional[str] = None  # category label (e.g. "open", "closed")
    index_version: Optional[str] = None  # retriever.index_version if available
    model_id: Optional[str] = None  # generator.model_id if available
    prompt_id: Optional[str] = None  # config.prompt_id if available


@dataclass
class Provenance:
    """The raw LLM answer that produced a result, plus all supporting evidence."""

    answer: str
    evidences: List[Evidence] = field(default_factory=list)
