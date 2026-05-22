"""
Zero-shot NLI-based classification baseline.

Implements the Yin et al. (2019) recipe: for each candidate label, frame
classification as a textual-entailment problem and ask a pretrained NLI
model whether the paper's premise (title + abstract) entails a hypothesis
that describes the label. The label with the highest entailment probability
wins for single-label tasks; for multi-label tasks, labels are kept whose
P(entailment) exceeds a configurable threshold.

This is the standard zero-shot classification baseline in modern NLP and
sits in the same design-space quadrant as EpiScope (label-free +
taxonomy-specifiable). Two hypothesis-construction modes are supported:

- ``label_name`` — Yin et al.'s canonical "This paper is about {label}."
  pattern, where ``{label}`` is the human-readable category name.
- ``label_description`` (default) — each template paragraph from the
  classifier config is used as a hypothesis directly. This gives NLI the
  same label specification text that EpiScope's prompt receives, making the
  comparison strictly fair on label-information access.

Entailment scores are cached on disk by ``(model, premise, hypothesis)`` so
repeated runs over the same evaluation sample do not recompute them.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from episcope.schemas import PaperMetadata
from classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
    default_labels,
    label_from_category,
    result_from_labels,
)
from episcope.workflows.classification.schemas import DataType, GeoRegion

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_NLI_CACHE_DIR = "outputs/.nli_cache"
DEFAULT_HYPOTHESIS_TEMPLATE = "This paper is about {label}."


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@dataclass
class NLIEntailmentScorer:
    """Frozen NLI model returning P(entailment) for (premise, hypothesis) pairs.

    Resolves the entailment/contradiction class indices from
    ``model.config.id2label`` so checkpoints with different label orderings
    (BART-MNLI vs DeBERTa-MNLI) work without manual mapping. Returns the
    Yin et al. score ``softmax([entailment_logit, contradiction_logit])[0]``
    — i.e. softmax over entailment and contradiction only, ignoring neutral.
    """

    model_name: str = DEFAULT_NLI_MODEL
    max_length: int = 512
    batch_size: int = 8
    device: Optional[str] = None
    cache_dir: Optional[str] = DEFAULT_NLI_CACHE_DIR

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._resolved_device: Optional[str] = None
        self._entail_idx: Optional[int] = None
        self._contradict_idx: Optional[int] = None
        self._memory_cache: dict[str, float] = {}
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self._resolved_device)

        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        for idx, label in id2label.items():
            if "entail" in label:
                self._entail_idx = idx
            elif "contradict" in label:
                self._contradict_idx = idx
        if self._entail_idx is None or self._contradict_idx is None:
            raise RuntimeError(
                f"Could not resolve entailment/contradiction class indices from "
                f"{self.model_name}; id2label={id2label}. Expected an NLI checkpoint."
            )

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        payload = f"{self.model_name}|{self.max_length}|{premise}|{hypothesis}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return Path(self.cache_dir) / f"{key}.npy"

    def _load_cached(self, key: str) -> Optional[float]:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._disk_path(key)
        if path is not None and path.exists():
            value = float(np.load(path))
            self._memory_cache[key] = value
            return value
        return None

    def _save_cached(self, key: str, value: float) -> None:
        self._memory_cache[key] = value
        path = self._disk_path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, np.float32(value))
            os.replace(tmp, path)

    def score(self, premise: str, hypotheses: Sequence[str]) -> np.ndarray:
        """Return ``P(entailment)`` per hypothesis for the given premise.

        Results are cached on disk by ``(model, premise, hypothesis)``.
        """
        import torch

        out = np.zeros(len(hypotheses), dtype=np.float32)
        pending: list[tuple[int, str, str]] = []  # (out_idx, hypothesis, cache_key)
        for idx, hypothesis in enumerate(hypotheses):
            key = self._cache_key(premise, hypothesis)
            cached = self._load_cached(key)
            if cached is not None:
                out[idx] = cached
            else:
                pending.append((idx, hypothesis, key))

        if not pending:
            return out

        self._load()
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            inputs = self._tokenizer(
                [premise] * len(batch),
                [hyp for _, hyp, _ in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._resolved_device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            # Yin et al.: softmax over (entailment, contradiction), drop neutral.
            entail_contradict = logits[:, [self._entail_idx, self._contradict_idx]]
            probs = torch.softmax(entail_contradict, dim=-1)[:, 0].detach().cpu().numpy()
            for (idx, _, key), value in zip(batch, probs):
                value_f = float(value)
                self._save_cached(key, value_f)
                out[idx] = value_f
        return out


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def _humanize_category(category: str, *, category_labels: Mapping[str, str]) -> str:
    """Pick the most readable form of a category for the label_name template."""
    if category in category_labels:
        label = category_labels[category]
        if label:
            return str(label).replace("_", " ").lower()
    return str(category).replace("_", " ").lower()


@dataclass
class _LabelHypothesis:
    """One (label, hypothesis-text) pair the scorer will evaluate."""

    label: Any
    text: str


class NLIZeroShotBaseline:
    """Zero-shot classification via NLI entailment scoring.

    Parameters
    ----------
    classifier_kind:
        One of ``paper_type``, ``data_accessibility``, ``data_type``, ``geo``.
    scorer:
        A configured ``NLIEntailmentScorer``.
    hypothesis_source:
        ``label_description`` (default) — use template paragraphs from the
        classifier config as hypotheses. ``label_name`` — use the canonical
        ``"This paper is about {label}."`` template.
    hypothesis_template:
        Template used when ``hypothesis_source == "label_name"``. Must contain
        ``{label}``.
    threshold:
        For multi-label tasks, keep labels whose ``P(entailment)`` exceeds
        this value. If no label clears the bar, the argmax label is kept.
    """

    name = "nli_zero_shot"

    def __init__(
        self,
        *,
        classifier_kind: str,
        scorer: NLIEntailmentScorer,
        hypothesis_source: str = "label_description",
        hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
        threshold: float = 0.5,
    ) -> None:
        if hypothesis_source not in ("label_name", "label_description"):
            raise ValueError(
                f"hypothesis_source must be 'label_name' or 'label_description'; "
                f"got {hypothesis_source!r}."
            )
        if hypothesis_source == "label_name" and "{label}" not in hypothesis_template:
            raise ValueError(
                "hypothesis_template must contain '{label}' when "
                "hypothesis_source='label_name'."
            )
        self.classifier_kind = classifier_kind
        self.config = classifier_config(classifier_kind)
        self.scorer = scorer
        self.hypothesis_source = hypothesis_source
        self.hypothesis_template = hypothesis_template
        self.threshold = float(threshold)
        self._hypotheses: list[_LabelHypothesis] = []
        self._build_hypotheses()

    def _build_hypotheses(self) -> None:
        category_labels = getattr(self.config, "category_labels", {}) or {}
        for category, paragraphs in self.config.template_paragraphs.items():
            label = label_from_category(self.classifier_kind, category)
            if label is None:
                continue
            if self.hypothesis_source == "label_name":
                readable = _humanize_category(category, category_labels=category_labels)
                self._hypotheses.append(
                    _LabelHypothesis(
                        label=label,
                        text=self.hypothesis_template.format(label=readable),
                    )
                )
            else:  # label_description: one hypothesis per template paragraph
                for paragraph in paragraphs:
                    paragraph = (paragraph or "").strip()
                    if paragraph:
                        self._hypotheses.append(
                            _LabelHypothesis(label=label, text=paragraph)
                        )

    @property
    def is_supported(self) -> bool:
        return bool(self._hypotheses)

    def _premise(self, metadata: PaperMetadata) -> str:
        title = (metadata.title or "").strip()
        abstract = (metadata.abstract or "").strip()
        if title and abstract:
            return f"{title} [SEP] {abstract}"
        return title or abstract

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        if not self.is_supported:
            result = result_from_labels(
                self.classifier_kind,
                default_labels(self.classifier_kind),
                confidence=0.0,
                reasoning=(
                    "NLI zero-shot baseline is not defined for this task because "
                    "its templates are not label-specific."
                ),
            )
            return BaselinePrediction(result=result)

        premise = self._premise(metadata)
        if not premise:
            result = result_from_labels(
                self.classifier_kind,
                default_labels(self.classifier_kind),
                confidence=0.0,
                reasoning="NLI zero-shot baseline had no premise text to score.",
            )
            return BaselinePrediction(result=result)

        scores = self.scorer.score(premise, [hyp.text for hyp in self._hypotheses])
        label_scores: dict[Any, float] = {}
        for hyp, score in zip(self._hypotheses, scores):
            label_scores[hyp.label] = max(label_scores.get(hyp.label, 0.0), float(score))

        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        if self.classifier_kind == "paper_type":
            labels = [ranked[0][0]]
            confidence = ranked[0][1]
        else:
            selected = [label for label, score in ranked if score >= self.threshold]
            labels = selected or [ranked[0][0]]
            confidence = ranked[0][1]

        if self.classifier_kind == "data_type" and DataType.NO_EMPIRICAL_DATA in labels:
            labels = [DataType.NO_EMPIRICAL_DATA]
        if self.classifier_kind == "geo" and GeoRegion.IRRELEVANT in labels:
            labels = [GeoRegion.IRRELEVANT]

        probabilities = {
            getattr(label, "name", str(label)): float(score) for label, score in ranked
        }
        result = result_from_labels(
            self.classifier_kind,
            labels,
            confidence=float(confidence),
            reasoning=(
                f"NLI zero-shot baseline ({self.scorer.model_name}, "
                f"source={self.hypothesis_source}): each candidate label was scored "
                f"as a textual-entailment hypothesis against the paper's title + "
                f"abstract; labels are kept by argmax (paper_type) or P(entailment) "
                f">= {self.threshold} (multi-label)."
            ),
            class_probabilities=probabilities,
            extras={
                "nli_model": self.scorer.model_name,
                "hypothesis_source": self.hypothesis_source,
                "threshold": self.threshold,
                "n_hypotheses": len(self._hypotheses),
                "top_scores": [
                    {"label": getattr(label, "name", str(label)), "score": float(score)}
                    for label, score in ranked[:5]
                ],
            },
        )
        return BaselinePrediction(result=result)
