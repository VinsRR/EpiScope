from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from classification.baselines.common import metadata_from_mapping, metadata_text
from classification.baselines.topic_models import (
    _build_bertopic_model,
    _component_terms,
    _effective_n_topics,
    _topic_matrix_from_assignments,
)


@dataclass(frozen=True)
class TopicExplorer:
    """Fit topic models for exploratory document clustering.

    Unlike the classification baselines, this class does not need gold labels or
    classifier configs. It is meant for EDA: discover latent themes, inspect
    clusters, and export document-topic assignments as data frames.
    """

    model_kind: str = "lsa"
    n_topics: int = 10
    max_features: int = 5000
    min_df: int | float = 1
    max_df: int | float = 0.95
    random_state: int = 13
    seed_topic_list: list[list[str]] | None = None

    def fit_records(self, records: Sequence[Mapping[str, Any]]) -> "FittedTopicExplorer":
        ids = [str(record.get("paper_id") or idx) for idx, record in enumerate(records)]
        texts = [
            metadata_text(metadata_from_mapping(record), record)
            or str(record.get("paper_id") or "empty document")
            for record in records
        ]
        return self.fit_texts(texts, ids=ids)

    def fit_texts(
        self,
        texts: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        labels: Sequence[int] | None = None,
    ) -> "FittedTopicExplorer":
        documents = [str(text or "") for text in texts]
        document_ids = (
            [str(item) for item in ids]
            if ids is not None
            else [str(idx) for idx in range(len(documents))]
        )
        if len(document_ids) != len(documents):
            raise ValueError("ids must have the same length as texts.")

        doc_topic, terms = self._fit_doc_topic(documents, labels=labels)
        topic_ids, topic_scores = self._assign_topics(doc_topic)
        return FittedTopicExplorer(
            config=self,
            document_ids=document_ids,
            documents=documents,
            doc_topic=doc_topic,
            topic_ids=topic_ids,
            topic_scores=topic_scores,
            topic_terms=terms,
        )

    def _fit_doc_topic(
        self,
        documents: Sequence[str],
        *,
        labels: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        model_kind = self.model_kind.lower()
        if model_kind == "lsa":
            vectorizer = self._tfidf_vectorizer()
            matrix = vectorizer.fit_transform(documents)
            n_components = _effective_n_topics(self.n_topics, matrix.shape[0], matrix.shape[1])
            model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            return np.abs(model.fit_transform(matrix)), _component_terms(
                model.components_,
                vectorizer.get_feature_names_out(),
            )
        if model_kind == "nmf":
            vectorizer = self._tfidf_vectorizer()
            matrix = vectorizer.fit_transform(documents)
            n_components = _effective_n_topics(self.n_topics, matrix.shape[0], matrix.shape[1])
            init = "nndsvda" if n_components <= min(matrix.shape) else "random"
            model = NMF(n_components=n_components, init=init, random_state=self.random_state, max_iter=500)
            return model.fit_transform(matrix), _component_terms(
                model.components_,
                vectorizer.get_feature_names_out(),
            )
        if model_kind == "lda":
            matrix, feature_names = self._count_matrix(documents)
            n_components = _effective_n_topics(self.n_topics, matrix.shape[0], matrix.shape[1])
            model = LatentDirichletAllocation(
                n_components=n_components,
                random_state=self.random_state,
                learning_method="batch",
            )
            return model.fit_transform(matrix), _component_terms(model.components_, feature_names)
        if model_kind == "plsa":
            matrix, feature_names = self._count_matrix(documents)
            n_components = _effective_n_topics(self.n_topics, matrix.shape[0], matrix.shape[1])
            model = NMF(
                n_components=n_components,
                init="random",
                solver="mu",
                beta_loss="kullback-leibler",
                random_state=self.random_state,
                max_iter=700,
            )
            return model.fit_transform(matrix), _component_terms(model.components_, feature_names)
        if model_kind in {"bertopic", "bertopic_guided", "bertopic_semisupervised"}:
            topic_model = _build_bertopic_model(
                n_topics=self.n_topics,
                n_documents=len(documents),
                random_state=self.random_state,
                seed_topic_list=self.seed_topic_list if model_kind == "bertopic_guided" else None,
                verbose=True,
            )
            topics, probabilities = topic_model.fit_transform(
                list(documents),
                y=list(labels) if labels is not None else None,
            )
            doc_topic, topic_index = _topic_matrix_from_assignments(topics, probabilities)
            terms = {
                topic_index[int(topic)]: [word for word, _ in topic_model.get_topic(topic) or []][:10]
                for topic in topic_index
            }
            return doc_topic, terms
        raise ValueError(f"Unknown exploratory topic model {self.model_kind!r}.")

    def _tfidf_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )

    def _count_matrix(self, documents: Sequence[str]):
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform(documents)
        return matrix, vectorizer.get_feature_names_out()

    @staticmethod
    def _assign_topics(doc_topic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if doc_topic.size == 0:
            return np.zeros((doc_topic.shape[0],), dtype=int), np.zeros((doc_topic.shape[0],))
        doc_topic = np.nan_to_num(np.asarray(doc_topic, dtype=float), nan=0.0)
        normalized = normalize(np.maximum(doc_topic, 0.0), norm="l1", axis=1)
        topic_ids = np.argmax(normalized, axis=1)
        scores = normalized[np.arange(normalized.shape[0]), topic_ids]
        return topic_ids, scores


@dataclass(frozen=True)
class FittedTopicExplorer:
    config: TopicExplorer
    document_ids: list[str]
    documents: list[str]
    doc_topic: np.ndarray
    topic_ids: np.ndarray
    topic_scores: np.ndarray
    topic_terms: dict[int, list[str]]

    def document_topics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "paper_id": self.document_ids,
                "topic_id": self.topic_ids.astype(int),
                "topic_score": self.topic_scores.astype(float),
                "text": self.documents,
            }
        )

    def topics(self) -> pd.DataFrame:
        rows = []
        for topic_id in sorted(set(int(topic) for topic in self.topic_ids)):
            mask = self.topic_ids == topic_id
            rows.append(
                {
                    "topic_id": topic_id,
                    "n_documents": int(mask.sum()),
                    "mean_topic_score": float(np.mean(self.topic_scores[mask])) if mask.any() else 0.0,
                    "top_terms": self.topic_terms.get(topic_id, []),
                }
            )
        return pd.DataFrame(rows)

    def clusters(self) -> dict[int, list[str]]:
        clusters: dict[int, list[str]] = {}
        for paper_id, topic_id in zip(self.document_ids, self.topic_ids):
            clusters.setdefault(int(topic_id), []).append(paper_id)
        return clusters

    def document_topic_matrix(self) -> pd.DataFrame:
        columns = [f"topic_{idx}" for idx in range(self.doc_topic.shape[1])]
        return pd.DataFrame(self.doc_topic, index=self.document_ids, columns=columns)
