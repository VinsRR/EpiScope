from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.precision_miner.schemas import ExtractionResult


@dataclass
class ExtractionTrace:
    """Prompt/response trace for extraction runs."""

    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


@dataclass
class DetailedExtractionResult:
    """Detailed extraction result including provenance, trace, and selected chunks."""

    paper_id: str
    metadata: PaperMetadata
    result: ExtractionResult
    provenance: Provenance
    trace: ExtractionTrace
    relevant_chunks: List[SearchResult] = field(default_factory=list)
