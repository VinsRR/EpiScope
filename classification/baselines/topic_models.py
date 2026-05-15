from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
    TruncatedSVD,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from episcope.schemas import PaperMetadata
from classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
    default_labels,
    labels_from_names,
    majority_label_set,
    metadata_from_mapping,
    metadata_text,
    parse_label_names,
    result_from_labels,
)

SKLEARN_TOPIC_BASELINES = {"lsa", "plsa", "lda", "nmf"}

BERTOPIC_UNSUPERVISED_BASELINES = {"bertopic"}
BERTOPIC_ZERO_SHOT_BASELINES = {"bertopic_guided"}
BERTOPIC_SEMISUPERVISED_BASELINES = {"bertopic_semisupervised"}
BERTOPIC_BASELINES = (
    BERTOPIC_UNSUPERVISED_BASELINES
    | BERTOPIC_ZERO_SHOT_BASELINES
    | BERTOPIC_SEMISUPERVISED_BASELINES
)

OPTIONAL_TOPIC_BASELINES = BERTOPIC_BASELINES

# All methods dispatched to TopicModelBaseline — the implementation boundary.
TOPIC_MODEL_BASELINES = (*sorted(SKLEARN_TOPIC_BASELINES), *sorted(OPTIONAL_TOPIC_BASELINES))

# Semantic family groupings (used by runners).
UNSUPERVISED_TOPIC_BASELINES: tuple[str, ...] = (
    *sorted(SKLEARN_TOPIC_BASELINES),
    *sorted(BERTOPIC_UNSUPERVISED_BASELINES),
)
ZERO_SHOT_TOPIC_BASELINES: tuple[str, ...] = (
    *sorted(BERTOPIC_ZERO_SHOT_BASELINES),
)

logger = logging.getLogger(__name__)


def _effective_n_topics(requested: int, n_documents: int, n_features: int) -> int:
    upper = max(1, min(n_documents, n_features))
    return max(1, min(int(requested), upper))


def _safe_corpus(records: Sequence[Mapping[str, object]]) -> list[str]:
    corpus = []
    for record in records:
        text = metadata_text(metadata_from_mapping(record), record)
        if not text:
            text = str(record.get("paper_id") or "empty document")
        corpus.append(text)
    return corpus


def _label_set_key(label_names: Sequence[str]) -> str:
    return "|".join(
        sorted(str(label).upper() for label in label_names if str(label).strip())
    )


def _tokenize_seed_text(text: str) -> list[str]:
    stop = {
        "and",
        "are",
        "for",
        "from",
        "into",
        "only",
        "that",
        "the",
        "this",
        "used",
        "with",
        "without",
    }
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z_\-]{2,}", text):
        clean = token.lower().replace("_", " ").replace("-", " ").strip()
        if clean and clean not in stop and clean not in terms:
            terms.append(clean)
    return terms


def _seed_topic_list(classifier_kind: str, *, max_terms: int = 12) -> list[list[str]]:
    config = classifier_config(classifier_kind)
    templates = {
        str(key).lower(): value for key, value in (config.template_paragraphs or {}).items()
    }
    seeds: list[list[str]] = []
    for key, label in config.category_labels.items():
        label_text = str(label)
        if label_text.upper() == "UNCLEAR":
            continue
        candidates = [str(key), label_text]
        lookup_keys = {
            str(key).lower(),
            label_text.lower(),
            label_text.lower().replace(" ", "_"),
            label_text.lower().replace("_", " "),
        }
        for lookup in lookup_keys:
            candidates.extend(templates.get(lookup, [])[:2])
        terms = _tokenize_seed_text(" ".join(candidates))
        if terms:
            seeds.append(terms[:max_terms])
    return seeds


def _gold_label_sets_for_records(
    *,
    records: Sequence[Mapping[str, object]],
    ground_truth_records: Sequence[Mapping[str, object]],
    ground_truth_column: str,
) -> list[tuple[str, ...]]:
    truth_by_id = {
        str(record.get("paper_id")): record
        for record in ground_truth_records
        if record.get("paper_id") is not None
    }
    out: list[tuple[str, ...]] = []
    for record in records:
        truth = truth_by_id.get(str(record.get("paper_id")), record)
        out.append(parse_label_names(truth.get(ground_truth_column)))
    return out


def _topic_matrix_from_assignments(
    topics: Sequence[int],
    probabilities: Any,
) -> tuple[np.ndarray, dict[int, int]]:
    topic_ids = sorted({int(topic) for topic in topics})
    topic_index = {topic: idx for idx, topic in enumerate(topic_ids)}
    doc_topic = np.zeros((len(topics), len(topic_ids)), dtype=float)
    probability_array = np.asarray(probabilities) if probabilities is not None else None
    if (
        probability_array is not None
        and probability_array.ndim == 2
        and probability_array.shape[0] == len(topics)
        and probability_array.shape[1] == len(topic_ids)
    ):
        doc_topic = np.asarray(probability_array, dtype=float)
    else:
        for row_idx, topic in enumerate(topics):
            doc_topic[row_idx, topic_index[int(topic)]] = 1.0
    return doc_topic, topic_index


def _component_terms(
    components: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_terms: int = 10,
) -> dict[int, list[str]]:
    terms: dict[int, list[str]] = {}
    features = np.asarray(feature_names)
    for topic_idx, component in enumerate(np.asarray(components)):
        order = np.argsort(component)[::-1][:n_terms]
        terms[topic_idx] = [
            str(features[index]) for index in order if index < len(features)
        ]
    return terms


def _build_bertopic_model(
    *,
    n_topics: int,
    n_documents: int,
    random_state: int,
    seed_topic_list: list[list[str]] | None = None,
    supervised: bool = False,
    verbose: bool = True,
):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer

    if supervised:
        from bertopic.dimensionality import BaseDimensionalityReduction
        from sklearn.linear_model import LogisticRegression

        return BERTopic(
            umap_model=BaseDimensionalityReduction(),
            hdbscan_model=LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            verbose=verbose,
        )

    from hdbscan import HDBSCAN
    from umap import UMAP

    n_neighbors = max(2, min(10, max(2, n_documents - 1)))
    n_components = max(2, min(5, max(2, n_documents - 2)))
    min_cluster_size = max(2, min(10, max(2, n_documents // max(2, n_topics))))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=1,
    )
    return BERTopic(
        nr_topics=n_topics,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        seed_topic_list=seed_topic_list,
        verbose=verbose,
    )



class TopicModelBaseline:
    """Unsupervised topic model followed by topic-to-label majority mapping."""

    def __init__(
        self,
        *,
        classifier_kind: str,
        model_kind: str,
        records: Sequence[Mapping[str, object]],
        ground_truth_records: Sequence[Mapping[str, object]],
        ground_truth_column: str,
        n_topics: int = 10,
        max_features: int = 5000,
        min_df: int | float = 1,
        max_df: int | float = 0.95,
        random_state: int = 13,
        leave_one_out: bool = True,
        semisupervised_label_fraction: float = 0.5,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.model_kind = model_kind.lower()
        self.records = [dict(record) for record in records]
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.leave_one_out = leave_one_out
        self.semisupervised_label_fraction = float(semisupervised_label_fraction)
        self.ground_truth_records = [dict(record) for record in ground_truth_records]
        self.ground_truth_column = ground_truth_column
        self.gold_by_paper_id = {
            str(record.get("paper_id")): parse_label_names(
                record.get(ground_truth_column)
            )
            for record in ground_truth_records
            if record.get("paper_id") is not None
        }
        self.global_label_names = majority_label_set(self.gold_by_paper_id.values())
        self.topic_by_paper_id: dict[str, int] = {}
        self.topic_score_by_paper_id: dict[str, float] = {}
        self.topic_members: dict[int, list[str]] = defaultdict(list)
        self.topic_terms: dict[int, list[str]] = {}
        self._fit()

    @property
    def name(self) -> str:
        return self.model_kind

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        topic_id = self.topic_by_paper_id.get(str(paper_id), -1)
        topic_members = list(self.topic_members.get(topic_id, []))
        label_source = [
            self.gold_by_paper_id[member_id]
            for member_id in topic_members
            if member_id in self.gold_by_paper_id
            and (not self.leave_one_out or member_id != str(paper_id))
        ]
        label_names = majority_label_set(label_source) or self.global_label_names
        labels = labels_from_names(self.classifier_kind, label_names)
        if not labels:
            labels = default_labels(self.classifier_kind)

        support = Counter(tuple(labels) for labels in label_source)
        support_total = sum(support.values())
        support_count = support.get(tuple(label_names), 0)
        confidence = support_count / support_total if support_total else 0.0
        topic_score = self.topic_score_by_paper_id.get(str(paper_id), 0.0)

        result = result_from_labels(
            self.classifier_kind,
            labels,
            confidence=float(
                max(confidence, topic_score if not support_total else 0.0)
            ),
            reasoning=(
                f"{self.model_kind.upper()} baseline: assigned the paper to topic "
                f"{topic_id}, then predicted the majority gold label set among "
                "papers assigned to that topic."
            ),
            class_probabilities={
                "topic_label_support": float(confidence),
                "topic_assignment_score": float(topic_score),
            },
            extras={
                "topic_model": self.model_kind,
                "topic_model_family": (
                    "bertopic"
                    if self.model_kind in BERTOPIC_BASELINES
                    else self.model_kind
                ),
                "topic_model_mode": self._topic_model_mode,
                "topic_id": topic_id,
                "topic_size": len(topic_members),
                "topic_score": topic_score,
                "top_terms": self.topic_terms.get(topic_id, []),
                "leave_one_out": self.leave_one_out,
            },
        )
        return BaselinePrediction(result=result)

    def _fit(self) -> None:
        if self.model_kind == "lsa":
            doc_topic, terms = self._fit_lsa()
        elif self.model_kind == "plsa":
            doc_topic, terms = self._fit_plsa()
        elif self.model_kind == "lda":
            doc_topic, terms = self._fit_lda()
        elif self.model_kind == "nmf":
            doc_topic, terms = self._fit_nmf()
        elif self.model_kind in BERTOPIC_BASELINES:
            doc_topic, terms = self._fit_bertopic()
        else:
            raise ValueError(
                f"Unknown topic baseline {self.model_kind!r}. "
                f"Use one of {list(TOPIC_MODEL_BASELINES)}."
            )

        topic_ids, topic_scores = self._topic_assignments(doc_topic)
        for record, topic_id, topic_score in zip(self.records, topic_ids, topic_scores):
            paper_id = str(record.get("paper_id"))
            self.topic_by_paper_id[paper_id] = int(topic_id)
            self.topic_score_by_paper_id[paper_id] = float(topic_score)
            self.topic_members[int(topic_id)].append(paper_id)
        self.topic_terms = terms

    def _fit_lsa(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        corpus = _safe_corpus(self.records)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform(corpus)
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        doc_topic = np.abs(model.fit_transform(matrix))
        terms = _component_terms(model.components_, vectorizer.get_feature_names_out())
        return doc_topic, terms

    def _fit_lda(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        matrix, feature_names = self._count_matrix()
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        model = LatentDirichletAllocation(
            n_components=n_components,
            random_state=self.random_state,
            learning_method="batch",
        )
        doc_topic = model.fit_transform(matrix)
        terms = _component_terms(model.components_, feature_names)
        return doc_topic, terms

    def _fit_nmf(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        corpus = _safe_corpus(self.records)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform(corpus)
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        init = "nndsvda" if n_components <= min(matrix.shape) else "random"
        model = NMF(
            n_components=n_components,
            init=init,
            random_state=self.random_state,
            max_iter=500,
        )
        doc_topic = model.fit_transform(matrix)
        terms = _component_terms(model.components_, vectorizer.get_feature_names_out())
        return doc_topic, terms

    def _fit_plsa(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        matrix, feature_names = self._count_matrix()
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        model = NMF(
            n_components=n_components,
            init="random",
            solver="mu",
            beta_loss="kullback-leibler",
            random_state=self.random_state,
            max_iter=700,
        )
        doc_topic = model.fit_transform(matrix)
        terms = _component_terms(model.components_, feature_names)
        return doc_topic, terms

    @property
    def _topic_model_mode(self) -> str:
        if self.model_kind == "bertopic_guided":
            return "guided"
        if self.model_kind == "bertopic_semisupervised":
            return "semi_supervised"
        return "unsupervised"

    def _semisupervised_targets(self) -> tuple[list[int], dict[int, tuple[str, ...]]]:
        label_sets = _gold_label_sets_for_records(
            records=self.records,
            ground_truth_records=self.ground_truth_records,
            ground_truth_column=self.ground_truth_column,
        )
        label_to_id: dict[str, int] = {}
        id_to_labels: dict[int, tuple[str, ...]] = {}
        targets: list[int] = []
        for labels in label_sets:
            key = _label_set_key(labels)
            if not key:
                targets.append(-1)
                continue
            if key not in label_to_id:
                label_id = len(label_to_id)
                label_to_id[key] = label_id
                id_to_labels[label_id] = labels
            targets.append(label_to_id[key])

        fraction = max(0.0, min(1.0, self.semisupervised_label_fraction))
        if fraction < 1.0:
            known_indices = [idx for idx, label_id in enumerate(targets) if label_id >= 0]
            keep_count = int(round(len(known_indices) * fraction))
            rng = np.random.default_rng(self.random_state)
            keep = (
                set(rng.choice(known_indices, size=keep_count, replace=False).tolist())
                if keep_count
                else set()
            )
            targets = [
                label_id if idx in keep else -1
                for idx, label_id in enumerate(targets)
            ]
        return targets, id_to_labels

    def _fit_bertopic(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        try:
            import bertopic  # noqa: F401
            import hdbscan  # noqa: F401
            import umap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "BERTopic baseline requires the optional 'bertopic' package. "
                "Install it in this environment before running --baseline-kind bertopic."
            ) from exc

        corpus = _safe_corpus(self.records)
        seed_topics = (
            _seed_topic_list(self.classifier_kind)
            if self.model_kind == "bertopic_guided"
            else None
        )
        model = _build_bertopic_model(
            n_topics=self.n_topics,
            n_documents=len(corpus),
            random_state=self.random_state,
            seed_topic_list=seed_topics,
        )
        targets = None
        if self.model_kind == "bertopic_semisupervised":
            targets, _ = self._semisupervised_targets()
        topics, probabilities = model.fit_transform(corpus, y=targets)
        doc_topic, topic_index = _topic_matrix_from_assignments(topics, probabilities)
        terms = {
            topic_index[int(topic)]: [word for word, _ in model.get_topic(topic) or []][
                :10
            ]
            for topic in topic_index
        }
        return doc_topic, terms

    def _count_matrix(self) -> tuple[Any, np.ndarray]:
        corpus = _safe_corpus(self.records)
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform(corpus)
        return matrix, vectorizer.get_feature_names_out()

    @staticmethod
    def _topic_assignments(doc_topic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if doc_topic.size == 0:
            return np.zeros((doc_topic.shape[0],), dtype=int), np.zeros(
                (doc_topic.shape[0],)
            )
        doc_topic = np.nan_to_num(np.asarray(doc_topic, dtype=float), nan=0.0)
        normalized = normalize(np.maximum(doc_topic, 0.0), norm="l1", axis=1)
        scores = np.max(normalized, axis=1)
        topic_ids = np.argmax(normalized, axis=1)
        return topic_ids, scores
