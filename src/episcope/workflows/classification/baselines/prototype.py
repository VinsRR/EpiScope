from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any

from episcope.schemas import PaperMetadata
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
    default_labels,
    label_from_category,
    metadata_from_mapping,
    metadata_text,
    result_from_labels,
)
from episcope.workflows.classification.schemas import DataType, GeoRegion


class PrototypeSimilarityBaseline:
    """TF-IDF nearest-prototype classifier using config template paragraphs."""

    name = "prototype_similarity"

    def __init__(
        self,
        *,
        classifier_kind: str,
        records: Sequence[Mapping[str, object]] = (),
        multilabel_ratio: float = 0.92,
        min_score: float = 0.03,
        max_labels: int = 3,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.config = classifier_config(classifier_kind)
        self.multilabel_ratio = multilabel_ratio
        self.min_score = min_score
        self.max_labels = max_labels
        self.prototype_texts: list[str] = []
        self.prototype_labels: list[Any] = []
        self._build_prototypes()
        if not self.prototype_texts:
            self.vectorizer = None
            self.prototype_matrix = None
            return
        corpus = [
            metadata_text(metadata_from_mapping(record), record)
            for record in records
            if metadata_text(metadata_from_mapping(record), record)
        ]
        fit_corpus = [*self.prototype_texts, *corpus] or [""]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.vectorizer.fit(fit_corpus)
        self.prototype_matrix = self.vectorizer.transform(self.prototype_texts)

    def _build_prototypes(self) -> None:
        for category, paragraphs in self.config.template_paragraphs.items():
            label = label_from_category(self.classifier_kind, category)
            if label is None:
                continue
            for paragraph in paragraphs:
                self.prototype_texts.append(paragraph)
                self.prototype_labels.append(label)

    @property
    def is_supported(self) -> bool:
        return bool(self.prototype_texts)

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
                    "Prototype-similarity baseline is not defined for this task "
                    "because its templates are not label-specific."
                ),
            )
            return BaselinePrediction(result=result)

        text = metadata_text(metadata, record)
        if not text:
            result = result_from_labels(
                self.classifier_kind,
                default_labels(self.classifier_kind),
                confidence=0.0,
                reasoning="Prototype-similarity baseline had no metadata text to score.",
            )
            return BaselinePrediction(result=result)

        doc_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(doc_vector, self.prototype_matrix)[0]
        label_scores: dict[Any, float] = {}
        for label, score in zip(self.prototype_labels, similarities):
            label_scores[label] = max(label_scores.get(label, 0.0), float(score))

        ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            labels = default_labels(self.classifier_kind)
            confidence = 0.0
        elif self.classifier_kind == "paper_type":
            labels = [ranked[0][0]]
            confidence = ranked[0][1]
        else:
            max_score = ranked[0][1]
            selected = [
                label
                for label, score in ranked
                if score >= self.min_score
                and score >= max_score * self.multilabel_ratio
            ][: self.max_labels]
            labels = selected or [ranked[0][0]]
            confidence = max_score

        if self.classifier_kind == "data_type" and DataType.NO_EMPIRICAL_DATA in labels:
            labels = [DataType.NO_EMPIRICAL_DATA]
        if self.classifier_kind == "geo" and GeoRegion.IRRELEVANT in labels:
            labels = [GeoRegion.IRRELEVANT]

        total = float(np.sum([score for _, score in ranked])) or 1.0
        probabilities = {
            getattr(label, "name", str(label)): score / total for label, score in ranked
        }
        result = result_from_labels(
            self.classifier_kind,
            labels,
            confidence=float(confidence),
            reasoning=(
                "Prototype-similarity baseline: selected labels whose metadata "
                "TF-IDF representation was closest to label template paragraphs."
            ),
            class_probabilities=probabilities,
            extras={
                "top_scores": [
                    {"label": getattr(label, "name", str(label)), "score": score}
                    for label, score in ranked[:5]
                ]
            },
        )
        return BaselinePrediction(result=result)
