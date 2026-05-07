from __future__ import annotations

import contextlib
import io
import logging
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
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    default_labels,
    labels_from_names,
    majority_label_set,
    metadata_from_mapping,
    metadata_text,
    parse_label_names,
    result_from_labels,
)

SKLEARN_TOPIC_BASELINES = {"lsa", "plsa", "lda", "nmf"}
OPTIONAL_TOPIC_BASELINES = {"bertopic", "top2vec"}
TOPIC_MODEL_BASELINES = (*sorted(SKLEARN_TOPIC_BASELINES), *sorted(OPTIONAL_TOPIC_BASELINES))

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
    verbose: bool = True,
):
    from bertopic import BERTopic
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
        verbose=verbose,
    )


def _top2vec_kwargs(*, n_documents: int, n_topics: int) -> dict[str, Any]:
    n_neighbors = max(2, min(10, max(2, n_documents - 1)))
    n_components = max(2, min(5, max(2, n_documents - 2)))
    min_cluster_size = max(2, min(10, max(2, n_documents // max(2, n_topics))))
    return {
        "embedding_model": "doc2vec",
        "speed": "fast-learn",
        "workers": 1,
        "index_topics": False,
        "verbose": False,
        "umap_args": {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": 0.0,
            "metric": "cosine",
            "low_memory": True,
        },
        "hdbscan_args": {
            "min_cluster_size": min_cluster_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": False,
            "core_dist_n_jobs": 1,
        },
    }


def _build_top2vec_model(
    *,
    documents: Sequence[str],
    n_topics: int,
):
    from top2vec import Top2Vec

    logging.getLogger("top2vec").setLevel(logging.WARNING)
    kwargs = _top2vec_kwargs(n_documents=len(documents), n_topics=n_topics)
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            return Top2Vec(documents=list(documents), **kwargs)
        except TypeError as exc:
            logger.debug(
                "Top2Vec rejected lightweight kwargs, retrying with minimal kwargs: %s",
                exc,
            )
            fallback = {
                "embedding_model": "doc2vec",
                "speed": "fast-learn",
                "workers": 1,
            }
            return Top2Vec(documents=list(documents), **fallback)


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
        elif self.model_kind == "bertopic":
            doc_topic, terms = self._fit_bertopic()
        elif self.model_kind == "top2vec":
            doc_topic, terms = self._fit_top2vec()
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
        model = _build_bertopic_model(
            n_topics=self.n_topics,
            n_documents=len(corpus),
            random_state=self.random_state,
        )
        topics, probabilities = model.fit_transform(corpus)
        doc_topic, topic_index = _topic_matrix_from_assignments(topics, probabilities)
        terms = {
            topic_index[int(topic)]: [word for word, _ in model.get_topic(topic) or []][
                :10
            ]
            for topic in topic_index
        }
        return doc_topic, terms

    def _fit_top2vec(self) -> tuple[np.ndarray, dict[int, list[str]]]:
        try:
            from top2vec import Top2Vec
        except ImportError as exc:
            raise RuntimeError(
                "Top2Vec baseline requires the optional 'top2vec' package. "
                "Install it in this environment before running --baseline-kind top2vec."
            ) from exc

        corpus = _safe_corpus(self.records)
        model = _build_top2vec_model(documents=corpus, n_topics=self.n_topics)
        topic_nums, topic_scores, *_ = model.get_documents_topics(
            doc_ids=list(range(len(corpus))),
            num_topics=1,
        )
        topic_nums = np.asarray(topic_nums).reshape(-1)
        topic_scores = np.asarray(topic_scores).reshape(-1)
        topic_ids = sorted({int(topic) for topic in topic_nums})
        topic_index = {topic: idx for idx, topic in enumerate(topic_ids)}
        doc_topic = np.zeros((len(corpus), len(topic_ids)), dtype=float)
        for row_idx, (topic, score) in enumerate(zip(topic_nums, topic_scores)):
            doc_topic[row_idx, topic_index[int(topic)]] = float(score)
        terms = {}
        try:
            topic_words, _, topic_ids_out = model.get_topics()
            for words, topic_id in zip(topic_words, topic_ids_out):
                if int(topic_id) in topic_index:
                    terms[topic_index[int(topic_id)]] = [
                        str(word) for word in words[:10]
                    ]
        except Exception:
            terms = {}
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
