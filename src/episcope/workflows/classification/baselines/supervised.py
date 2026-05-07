from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.svm import LinearSVC

from episcope.schemas import PaperMetadata
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
    default_labels,
    labels_from_names,
    metadata_from_mapping,
    metadata_text,
    parse_label_names,
    result_from_labels,
)

SUPERVISED_CLASSIFIER_BASELINES = (
    "supervised_tfidf_logreg",
    "supervised_tfidf_linear_svm",
)
SUPERVISED_TOPIC_BASELINES = (
    "supervised_lsa_logreg",
    "supervised_plsa_logreg",
    "supervised_lda_logreg",
    "supervised_nmf_logreg",
)
SUPERVISED_BASELINES = (*SUPERVISED_CLASSIFIER_BASELINES, *SUPERVISED_TOPIC_BASELINES)


def is_supervised_baseline(baseline_kind: str) -> bool:
    return baseline_kind in SUPERVISED_BASELINES


def is_supervised_topic_baseline(baseline_kind: str) -> bool:
    return baseline_kind in SUPERVISED_TOPIC_BASELINES


def label_names_for_task(classifier_kind: str) -> tuple[str, ...]:
    labels = []
    for label in classifier_config(classifier_kind).classification_mapping.values():
        name = str(getattr(label, "name", label)).upper()
        if name == "UNCLEAR":
            continue
        if name not in labels:
            labels.append(name)
    return tuple(labels)


def _records_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(record.get("paper_id")): record
        for record in records
        if record.get("paper_id") is not None
    }


def _safe_corpus(records: Sequence[Mapping[str, Any]]) -> list[str]:
    corpus = []
    for record in records:
        text = metadata_text(metadata_from_mapping(record), record)
        corpus.append(text or str(record.get("paper_id") or "empty document"))
    return corpus


def _label_matrix(
    *,
    classifier_kind: str,
    records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    ground_truth_column: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    label_names = label_names_for_task(classifier_kind)
    label_index = {name: idx for idx, name in enumerate(label_names)}
    truth_by_id = _records_by_id(ground_truth_records)
    matrix = np.zeros((len(records), len(label_names)), dtype=int)
    for row_idx, record in enumerate(records):
        truth = truth_by_id.get(str(record.get("paper_id")), record)
        for name in parse_label_names(truth.get(ground_truth_column)):
            if name in label_index:
                matrix[row_idx, label_index[name]] = 1
    return matrix, label_names


def _cv_splits(n_samples: int, *, cv_mode: str, cv_folds: int, random_state: int):
    if n_samples < 2:
        raise ValueError("Supervised baselines require at least two labelled papers.")
    if cv_mode == "leave_one_out":
        return list(LeaveOneOut().split(np.arange(n_samples)))
    if cv_mode == "kfold":
        n_splits = max(2, min(int(cv_folds), n_samples))
        return list(
            KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
                np.arange(n_samples)
            )
        )
    raise ValueError("cv_mode must be one of: kfold, leave_one_out")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


class SupervisedCVBaseline:
    """Cross-validated supervised baseline over full paper text."""

    def __init__(
        self,
        *,
        classifier_kind: str,
        baseline_kind: str,
        records: Sequence[Mapping[str, Any]],
        ground_truth_records: Sequence[Mapping[str, Any]],
        ground_truth_column: str,
        cv_mode: str = "kfold",
        cv_folds: int = 5,
        n_topics: int = 10,
        max_features: int = 5000,
        min_df: int | float = 1,
        max_df: int | float = 0.95,
        random_state: int = 13,
        threshold: float = 0.5,
        max_iter: int = 1000,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.baseline_kind = baseline_kind
        self.records = [dict(record) for record in records]
        self.cv_mode = cv_mode
        self.cv_folds = cv_folds
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.threshold = threshold
        self.max_iter = max_iter
        self.paper_ids = [str(record.get("paper_id")) for record in self.records]
        self.y_true, self.label_names = _label_matrix(
            classifier_kind=classifier_kind,
            records=self.records,
            ground_truth_records=ground_truth_records,
            ground_truth_column=ground_truth_column,
        )
        self.predictions: dict[str, BaselinePrediction] = {}
        self._fit_predict()

    @property
    def name(self) -> str:
        return self.baseline_kind

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        prediction = self.predictions.get(str(paper_id))
        if prediction is not None:
            return prediction
        result = result_from_labels(
            self.classifier_kind,
            default_labels(self.classifier_kind),
            confidence=0.0,
            reasoning="Supervised CV baseline had no precomputed prediction for this paper.",
        )
        return BaselinePrediction(result=result)

    def _fit_predict(self) -> None:
        corpus = _safe_corpus(self.records)
        splits = _cv_splits(
            len(corpus),
            cv_mode=self.cv_mode,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
        )
        probabilities = np.zeros_like(self.y_true, dtype=float)
        fold_by_index: dict[int, int] = {}

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            train_texts = [corpus[idx] for idx in train_idx]
            test_texts = [corpus[idx] for idx in test_idx]
            x_train, x_test = self._features(train_texts, test_texts)
            y_train = self.y_true[train_idx]
            probabilities[test_idx] = self._fit_predict_probabilities(x_train, y_train, x_test)
            for idx in test_idx:
                fold_by_index[int(idx)] = fold_idx

        for row_idx, paper_id in enumerate(self.paper_ids):
            scores = probabilities[row_idx]
            selected_names = self._selected_label_names(scores)
            labels = labels_from_names(self.classifier_kind, selected_names)
            if not labels:
                labels = default_labels(self.classifier_kind)
            ranked = sorted(
                zip(self.label_names, scores),
                key=lambda item: item[1],
                reverse=True,
            )
            result = result_from_labels(
                self.classifier_kind,
                labels,
                confidence=float(np.max(scores)) if len(scores) else 0.0,
                reasoning=(
                    f"{self.baseline_kind} supervised baseline: trained on "
                    f"{self.cv_mode} folds and predicted this paper from its held-out fold."
                ),
                class_probabilities={
                    label: float(score) for label, score in zip(self.label_names, scores)
                },
                extras={
                    "cv_mode": self.cv_mode,
                    "cv_folds": self.cv_folds if self.cv_mode == "kfold" else len(self.records),
                    "fold_index": fold_by_index.get(row_idx),
                    "n_topics": self.n_topics if self._uses_topic_features else None,
                    "top_scores": [
                        {"label": label, "score": float(score)}
                        for label, score in ranked[:8]
                    ],
                },
            )
            self.predictions[paper_id] = BaselinePrediction(result=result)

    @property
    def _uses_topic_features(self) -> bool:
        return self.baseline_kind in SUPERVISED_TOPIC_BASELINES

    def _features(self, train_texts: Sequence[str], test_texts: Sequence[str]):
        if self.baseline_kind == "supervised_tfidf_logreg" or self.baseline_kind == "supervised_tfidf_linear_svm":
            vectorizer = self._tfidf_vectorizer()
            x_train = vectorizer.fit_transform(train_texts)
            x_test = vectorizer.transform(test_texts)
            return x_train, x_test
        if self.baseline_kind == "supervised_lsa_logreg":
            vectorizer = self._tfidf_vectorizer()
            matrix_train = vectorizer.fit_transform(train_texts)
            matrix_test = vectorizer.transform(test_texts)
            n_components = self._effective_components(matrix_train.shape)
            model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            return model.fit_transform(matrix_train), model.transform(matrix_test)
        if self.baseline_kind == "supervised_nmf_logreg":
            vectorizer = self._tfidf_vectorizer()
            matrix_train = vectorizer.fit_transform(train_texts)
            matrix_test = vectorizer.transform(test_texts)
            n_components = self._effective_components(matrix_train.shape)
            init = "nndsvda" if n_components <= min(matrix_train.shape) else "random"
            model = NMF(
                n_components=n_components,
                init=init,
                random_state=self.random_state,
                max_iter=500,
            )
            return model.fit_transform(matrix_train), model.transform(matrix_test)
        if self.baseline_kind == "supervised_lda_logreg":
            vectorizer = self._count_vectorizer()
            matrix_train = vectorizer.fit_transform(train_texts)
            matrix_test = vectorizer.transform(test_texts)
            n_components = self._effective_components(matrix_train.shape)
            model = LatentDirichletAllocation(
                n_components=n_components,
                random_state=self.random_state,
                learning_method="batch",
            )
            return model.fit_transform(matrix_train), model.transform(matrix_test)
        if self.baseline_kind == "supervised_plsa_logreg":
            vectorizer = self._count_vectorizer()
            matrix_train = vectorizer.fit_transform(train_texts)
            matrix_test = vectorizer.transform(test_texts)
            n_components = self._effective_components(matrix_train.shape)
            model = NMF(
                n_components=n_components,
                init="random",
                solver="mu",
                beta_loss="kullback-leibler",
                random_state=self.random_state,
                max_iter=700,
            )
            return model.fit_transform(matrix_train), model.transform(matrix_test)
        raise ValueError(f"Unknown supervised baseline {self.baseline_kind!r}.")

    def _fit_predict_probabilities(
        self,
        x_train: Any,
        y_train: np.ndarray,
        x_test: Any,
    ) -> np.ndarray:
        out = np.zeros((x_test.shape[0], len(self.label_names)), dtype=float)
        for label_idx in range(len(self.label_names)):
            y = y_train[:, label_idx]
            unique = np.unique(y)
            if len(unique) == 1:
                out[:, label_idx] = float(unique[0])
                continue
            if self.baseline_kind == "supervised_tfidf_linear_svm":
                model = LinearSVC(class_weight="balanced", random_state=self.random_state)
                model.fit(x_train, y)
                out[:, label_idx] = _sigmoid(model.decision_function(x_test))
            else:
                model = LogisticRegression(
                    class_weight="balanced",
                    max_iter=self.max_iter,
                    solver="liblinear",
                    random_state=self.random_state,
                )
                model.fit(x_train, y)
                out[:, label_idx] = model.predict_proba(x_test)[:, 1]
        return out

    def _selected_label_names(self, scores: np.ndarray) -> tuple[str, ...]:
        selected = [
            label
            for label, score in zip(self.label_names, scores)
            if float(score) >= self.threshold
        ]
        if selected:
            return tuple(selected)
        if len(scores) == 0:
            return ()
        return (self.label_names[int(np.argmax(scores))],)

    def _effective_components(self, shape: tuple[int, int]) -> int:
        n_samples, n_features = shape
        upper = max(1, min(n_samples, n_features))
        return max(1, min(int(self.n_topics), upper))

    def _tfidf_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )

    def _count_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
