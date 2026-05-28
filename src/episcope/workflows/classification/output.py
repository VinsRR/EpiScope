from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from episcope.rag.provenance import Provenance
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.schemas import ClassificationResult


@dataclass
class CompletionSample:
    """One (prompt, completion) pair from the retry loop."""

    messages: List[Dict[str, str]]
    completion: str
    parsed_ok: bool
    reward: Optional[float] = None


@dataclass
class ClassificationDecision:
    """Small app-facing classification result."""

    paper_id: str
    metadata: PaperMetadata
    result: ClassificationResult
    top_evidence: List[SearchResult] = field(default_factory=list)


@dataclass
class ClassificationTrace:
    """Prompt/response trace useful for debugging and inspection."""

    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


@dataclass
class ClassificationTrainingRecord:
    """Artifacts collected during classification for inspection and evaluation."""

    paper_id: str
    result: ClassificationResult
    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    raw_llm_response: Optional[str] = None
    all_samples: List[CompletionSample] = field(default_factory=list)
    gold_label: Optional[Any] = None

    @property
    def reward(self) -> Optional[float]:
        if self.gold_label is None or not self.result.classification:
            return None
        return float(self.result.classification == self.gold_label)


@dataclass
class DetailedClassificationResult:
    """Full classification result with decision, provenance, trace, and training artifacts."""

    decision: ClassificationDecision
    provenance: Provenance
    trace: ClassificationTrace
    training: ClassificationTrainingRecord
