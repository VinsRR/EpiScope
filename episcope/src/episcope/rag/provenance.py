from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.schemas import ClassificationResult


# ---------------------------------------------------------------------------
# Provenance / evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Full lineage record for one retrieved snippet."""
    paper_id: str
    snippet: str
    section: Optional[str] = None           # category label (e.g. "open", "closed")
    index_version: Optional[str] = None     # retriever.index_version if available
    model_id: Optional[str] = None          # generator.model_id if available
    prompt_id: Optional[str] = None         # config.prompt_id if available


@dataclass
class Provenance:
    """The raw LLM answer that produced a result, plus all supporting evidence."""
    answer: str
    evidences: List[Evidence] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class CompletionSample:
    """One (prompt, completion) pair from the retry loop.

    Collecting all attempts — not just the successful one — is what makes
    RL-based training possible: failed attempts are the low-reward samples
    that preference optimisers (DPO, GRPO, RLOO) need.
    """
    messages: List[Dict[str, str]]   # prompt only — no assistant turn
    completion: str                  # raw model output for this attempt
    parsed_ok: bool                  # did it parse into a valid ClassificationResult?
    reward: Optional[float] = None   # set externally by a verifier / curator


# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------

@dataclass
class ClassificationOutput:
    """Full trace of one PaperClassifier.run() call.

    Carries everything needed for downstream consumption, dataset construction,
    and both SFT and RL-based fine-tuning.
    """
    paper_id: str
    metadata: PaperMetadata
    result: ClassificationResult
    provenance: Provenance
    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    raw_llm_response: Optional[str] = None
    all_samples: List[CompletionSample] = field(default_factory=list)
    gold_label: Optional[Any] = None    # attach externally after curation

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @property
    def reward(self) -> Optional[float]:
        """Exact-match reward against gold_label. None if no label is set."""
        if self.gold_label is None or not self.result.classification:
            return None
        return float(self.result.classification == self.gold_label)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_finetune_sample(self) -> Optional[Dict[str, Any]]:
        """SFT format: single (prompt + winning completion) pair."""
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
        """RL format: one record per attempt, prompt and completion kept separate.

        Only includes samples that have a reward attached. Attach rewards
        externally (e.g. via gold_label + a verifier) before calling this.
        """
        return [
            {
                "prompt": s.messages,
                "completion": s.completion,
                "reward": s.reward,
                "parsed_ok": s.parsed_ok,
                "paper_id": self.paper_id,
            }
            for s in self.all_samples
            if s.reward is not None
        ]

    def to_dpo_pair(self) -> Optional[Dict[str, Any]]:
        """DPO format: one chosen + one rejected completion for the same prompt.

        Requires at least one parsed and one failed attempt in all_samples.
        """
        chosen = next((s for s in self.all_samples if s.parsed_ok), None)
        rejected = next((s for s in self.all_samples if not s.parsed_ok), None)
        if not chosen or not rejected:
            return None
        return {
            "prompt": chosen.messages,
            "chosen": chosen.completion,
            "rejected": rejected.completion,
            "paper_id": self.paper_id,
        }