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
    """Training-oriented artifacts collected during classification."""

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

    def to_finetune_sample(self) -> Optional[Dict[str, Any]]:
        if not self.prompt_messages or self.raw_llm_response is None:
            return None
        return {
            "messages": [
                *self.prompt_messages,
                {"role": "assistant", "content": self.raw_llm_response},
            ],
            "paper_id": self.paper_id,
            "label": self.result.classification,
        }

    def to_rl_samples(self) -> List[Dict[str, Any]]:
        return [
            {
                "prompt": sample.messages,
                "completion": sample.completion,
                "reward": sample.reward,
                "parsed_ok": sample.parsed_ok,
                "paper_id": self.paper_id,
            }
            for sample in self.all_samples
            if sample.reward is not None
        ]

    def to_dpo_pair(self) -> Optional[Dict[str, Any]]:
        chosen = next((sample for sample in self.all_samples if sample.parsed_ok), None)
        rejected = next((sample for sample in self.all_samples if not sample.parsed_ok), None)
        if not chosen or not rejected:
            return None
        return {
            "prompt": chosen.messages,
            "chosen": chosen.completion,
            "rejected": rejected.completion,
            "paper_id": self.paper_id,
        }


@dataclass
class DetailedClassificationResult:
    """Full classification result with decision, provenance, trace, and training artifacts."""

    decision: ClassificationDecision
    provenance: Provenance
    trace: ClassificationTrace
    training: ClassificationTrainingRecord
