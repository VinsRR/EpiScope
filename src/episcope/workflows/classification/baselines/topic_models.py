from __future__ import annotations

import contextlib
import io
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
    TruncatedSVD,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from episcope.schemas import PaperMetadata
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
    default_labels,
    label_from_category,
    labels_from_names,
    majority_label_set,
    metadata_from_mapping,
    metadata_text,
    parse_label_names,
    result_from_labels,
)
from episcope.workflows.classification.schemas import (
    DATA_ACCESS_CODE_DEFINITIONS,
    DATA_TYPE_CODE_DEFINITIONS,
    GEO_CODE_DEFINITIONS,
    PAPER_TYPE_DEFINITIONS,
)

SKLEARN_TOPIC_BASELINES = {"lsa", "plsa", "lda", "nmf"}
OPTIONAL_TOPIC_BASELINES = {"bertopic", "top2vec"}
BASE_TOPIC_MODEL_KINDS = (
    *sorted(SKLEARN_TOPIC_BASELINES),
    *sorted(OPTIONAL_TOPIC_BASELINES),
)
GUIDED_TOPIC_MODEL_BASELINES = tuple(
    f"guided_{kind}" for kind in BASE_TOPIC_MODEL_KINDS
)
TOPIC_MAJORITY_BASELINES = tuple(
    f"topic_majority_{kind}" for kind in BASE_TOPIC_MODEL_KINDS
)
TOPIC_MODEL_BASELINES = GUIDED_TOPIC_MODEL_BASELINES

DEFINITIONS_BY_CLASSIFIER = {
    "paper_type": PAPER_TYPE_DEFINITIONS,
    "data_accessibility": DATA_ACCESS_CODE_DEFINITIONS,
    "data_type": DATA_TYPE_CODE_DEFINITIONS,
    "geo": GEO_CODE_DEFINITIONS,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelGuide:
    label: Any
    name: str
    text: str


def base_topic_model_kind(model_kind: str) -> str:
    kind = model_kind.lower()
    for prefix in ("guided_", "topic_majority_"):
        if kind.startswith(prefix):
            return kind.removeprefix(prefix)
    return kind


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


def _unique_text_parts(parts: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _label_guides(classifier_kind: str) -> list[LabelGuide]:
    config = classifier_config(classifier_kind)
    definitions = DEFINITIONS_BY_CLASSIFIER.get(classifier_kind, {})
    by_label: dict[Any, list[str]] = {}

    def add(label: Any, *parts: Any) -> None:
        if getattr(label, "name", "") == "UNCLEAR":
            return
        by_label.setdefault(label, []).extend(parts)

    for key, label in config.classification_mapping.items():
        label_name = getattr(label, "name", str(label))
        label_value = getattr(label, "value", label_name)
        category_label = (
            config.category_labels.get(key)
            or config.category_labels.get(str(label_value))
            or label_name
        )
        definition = (
            definitions.get(key)
            or definitions.get(label_name)
            or definitions.get(str(label_value))
            or ""
        )
        add(
            label,
            f"Classifier label {label_name}: {label_value}.",
            category_label,
            definition,
        )

    for category, paragraphs in config.template_paragraphs.items():
        label = label_from_category(classifier_kind, category)
        if label is None:
            continue
        add(label, category, *paragraphs)

    if classifier_kind == "geo":
        for label, parts in list(by_label.items()):
            name = str(getattr(label, "name", label)).replace("_", " ").title()
            parts.append(f"Epidemiological evidence geographically located in {name}.")

    guides = []
    for label, parts in by_label.items():
        label_name = str(getattr(label, "name", label))
        text = "\n".join(_unique_text_parts(parts))
        if text:
            guides.append(LabelGuide(label=label, name=label_name, text=text))
    return guides


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
        self.model_kind = base_topic_model_kind(model_kind)
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
                f"Use one of {list(BASE_TOPIC_MODEL_KINDS)}."
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


class GuidedTopicModelBaseline:
    """Topic/embedding baseline guided by task label descriptions, not gold labels."""

    def __init__(
        self,
        *,
        classifier_kind: str,
        model_kind: str,
        records: Sequence[Mapping[str, object]],
        n_topics: int = 10,
        max_features: int = 5000,
        min_df: int | float = 1,
        max_df: int | float = 0.95,
        random_state: int = 13,
        multilabel_ratio: float = 0.6,
        min_score: float = 0.05,
        max_labels: int = 4,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.model_kind = base_topic_model_kind(model_kind)
        self.records = [dict(record) for record in records]
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.multilabel_ratio = multilabel_ratio
        self.min_score = min_score
        self.max_labels = max_labels
        self.guides = _label_guides(classifier_kind)
        if not self.guides:
            raise ValueError(f"No label guides are configured for {classifier_kind!r}.")

        self.topic_terms: dict[int, list[str]] = {}
        self.score_by_paper_id: dict[str, dict[str, float]] = {}
        self.top_scores_by_paper_id: dict[str, list[dict[str, Any]]] = {}
        self._fit()

    @property
    def name(self) -> str:
        return f"guided_{self.model_kind}"

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        scores = self.score_by_paper_id.get(str(paper_id), {})
        ranked = sorted(
            ((guide, scores.get(guide.name, 0.0)) for guide in self.guides),
            key=lambda item: item[1],
            reverse=True,
        )
        selected = self._select_labels(ranked)
        confidence = float(ranked[0][1]) if ranked else 0.0

        total = float(sum(max(score, 0.0) for _, score in ranked)) or 1.0
        probabilities = {
            guide.name: float(max(score, 0.0) / total) for guide, score in ranked
        }
        extras: dict[str, Any] = {
            "baseline_mode": "guided_topic_model",
            "topic_model": self.model_kind,
            "top_scores": self.top_scores_by_paper_id.get(str(paper_id), []),
            "n_topics": self.n_topics,
            "multilabel_ratio": self.multilabel_ratio,
            "min_score": self.min_score,
            "max_labels": self._max_labels_for_task(),
        }
        if self.classifier_kind == "paper_type" and selected:
            extras["primary_label"] = selected[0]
            extras["secondary_labels"] = selected[1:]

        result = result_from_labels(
            self.classifier_kind,
            selected,
            confidence=confidence,
            reasoning=(
                f"Guided {self.model_kind.upper()} baseline: projected the paper and "
                "label-definition prompts into the same topic/embedding space, then "
                "selected every label close to the best-scoring label."
            ),
            class_probabilities=probabilities,
            extras=extras,
        )
        return BaselinePrediction(result=result)

    def _fit(self) -> None:
        corpus = _safe_corpus(self.records)
        guide_texts = [guide.text for guide in self.guides]
        all_embeddings, terms = self._fit_embeddings(corpus, guide_texts)
        n_documents = len(corpus)
        doc_embeddings = all_embeddings[:n_documents]
        guide_embeddings = all_embeddings[n_documents:]

        similarities = cosine_similarity(doc_embeddings, guide_embeddings)
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        if similarities.size and np.min(similarities) < 0:
            similarities = (similarities + 1.0) / 2.0
        similarities = np.clip(similarities, 0.0, 1.0)

        for row_idx, record in enumerate(self.records):
            paper_id = str(record.get("paper_id"))
            label_scores = {
                guide.name: float(similarities[row_idx, guide_idx])
                for guide_idx, guide in enumerate(self.guides)
            }
            ranked = sorted(
                label_scores.items(), key=lambda item: item[1], reverse=True
            )
            self.score_by_paper_id[paper_id] = label_scores
            self.top_scores_by_paper_id[paper_id] = [
                {"label": label, "score": score} for label, score in ranked[:8]
            ]
        self.topic_terms = terms

    def _fit_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        if self.model_kind == "lsa":
            return self._fit_lsa_embeddings(corpus, guide_texts)
        if self.model_kind == "plsa":
            return self._fit_plsa_embeddings(corpus, guide_texts)
        if self.model_kind == "lda":
            return self._fit_lda_embeddings(corpus, guide_texts)
        if self.model_kind == "nmf":
            return self._fit_nmf_embeddings(corpus, guide_texts)
        if self.model_kind == "bertopic":
            return self._fit_bertopic_embeddings(corpus, guide_texts)
        if self.model_kind == "top2vec":
            return self._fit_top2vec_embeddings(corpus, guide_texts)
        raise ValueError(
            f"Unknown guided topic baseline {self.model_kind!r}. "
            f"Use one of {list(BASE_TOPIC_MODEL_KINDS)}."
        )

    def _fit_lsa_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform([*corpus, *guide_texts])
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        embeddings = model.fit_transform(matrix)
        terms = _component_terms(model.components_, vectorizer.get_feature_names_out())
        return embeddings, terms

    def _fit_lda_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        matrix, feature_names = self._count_matrix([*corpus, *guide_texts])
        n_components = _effective_n_topics(
            self.n_topics, matrix.shape[0], matrix.shape[1]
        )
        model = LatentDirichletAllocation(
            n_components=n_components,
            random_state=self.random_state,
            learning_method="batch",
        )
        embeddings = model.fit_transform(matrix)
        terms = _component_terms(model.components_, feature_names)
        return embeddings, terms

    def _fit_nmf_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform([*corpus, *guide_texts])
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
        embeddings = model.fit_transform(matrix)
        terms = _component_terms(model.components_, vectorizer.get_feature_names_out())
        return embeddings, terms

    def _fit_plsa_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        matrix, feature_names = self._count_matrix([*corpus, *guide_texts])
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
        embeddings = model.fit_transform(matrix)
        terms = _component_terms(model.components_, feature_names)
        return embeddings, terms

    def _fit_bertopic_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        try:
            import bertopic  # noqa: F401
            import hdbscan  # noqa: F401
            import umap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "BERTopic baseline requires the optional 'bertopic' package. "
                "Install it in this environment before running --baseline-kind guided_bertopic."
            ) from exc

        all_texts = [*corpus, *guide_texts]
        model = _build_bertopic_model(
            n_topics=self.n_topics,
            n_documents=len(all_texts),
            random_state=self.random_state,
        )
        topics, probabilities = model.fit_transform(all_texts)
        embeddings, topic_index = _topic_matrix_from_assignments(topics, probabilities)
        terms = {
            topic_index[int(topic)]: [word for word, _ in model.get_topic(topic) or []][
                :10
            ]
            for topic in topic_index
        }
        return embeddings, terms

    def _fit_top2vec_embeddings(
        self,
        corpus: Sequence[str],
        guide_texts: Sequence[str],
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        try:
            from top2vec import Top2Vec
        except ImportError as exc:
            raise RuntimeError(
                "Top2Vec baseline requires the optional 'top2vec' package. "
                "Install it in this environment before running --baseline-kind guided_top2vec."
            ) from exc

        all_texts = [*corpus, *guide_texts]
        model = _build_top2vec_model(documents=all_texts, n_topics=self.n_topics)
        try:
            n_topics = max(1, min(self.n_topics, int(model.get_num_topics())))
        except Exception:
            n_topics = 1
        topic_nums, topic_scores, *_ = model.get_documents_topics(
            doc_ids=list(range(len(all_texts))),
            num_topics=n_topics,
        )
        topic_nums = np.asarray(topic_nums)
        topic_scores = np.asarray(topic_scores)
        if topic_nums.ndim == 1:
            topic_nums = topic_nums.reshape(-1, 1)
            topic_scores = topic_scores.reshape(-1, 1)

        topic_ids = sorted({int(topic) for topic in topic_nums.reshape(-1)})
        topic_index = {topic: idx for idx, topic in enumerate(topic_ids)}
        embeddings = np.zeros((len(all_texts), len(topic_ids)), dtype=float)
        for row_idx in range(topic_nums.shape[0]):
            for col_idx in range(topic_nums.shape[1]):
                topic = int(topic_nums[row_idx, col_idx])
                embeddings[row_idx, topic_index[topic]] = max(
                    embeddings[row_idx, topic_index[topic]],
                    float(topic_scores[row_idx, col_idx]),
                )

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
        return embeddings, terms

    def _count_matrix(self, texts: Sequence[str]) -> tuple[Any, np.ndarray]:
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        matrix = vectorizer.fit_transform(texts)
        return matrix, vectorizer.get_feature_names_out()

    def _select_labels(self, ranked: Sequence[tuple[LabelGuide, float]]) -> list[Any]:
        if not ranked:
            return default_labels(self.classifier_kind)

        best_score = ranked[0][1]
        selected = [
            guide.label
            for guide, score in ranked
            if score >= self.min_score and score >= best_score * self.multilabel_ratio
        ][: self._max_labels_for_task()]
        if not selected:
            selected = [ranked[0][0].label]
        return self._apply_exclusive_labels(selected)

    def _max_labels_for_task(self) -> int:
        if self.classifier_kind == "geo":
            return max(self.max_labels, 6)
        return self.max_labels

    def _apply_exclusive_labels(self, labels: Sequence[Any]) -> list[Any]:
        if not labels:
            return default_labels(self.classifier_kind)
        top = labels[0]
        top_name = str(getattr(top, "name", top))
        names_to_drop: set[str] = {"UNCLEAR"}
        exclusive_names: set[str] = set()
        if self.classifier_kind == "paper_type":
            exclusive_names = {"OTHER", "UNCLEAR"}
        elif self.classifier_kind == "data_accessibility":
            exclusive_names = {"NOT_STATED", "UNCLEAR"}
        elif self.classifier_kind == "data_type":
            exclusive_names = {"NO_EMPIRICAL_DATA", "UNCLEAR"}
        elif self.classifier_kind == "geo":
            exclusive_names = {"IRRELEVANT", "UNCLEAR"}

        if top_name in exclusive_names:
            return [top]

        names_to_drop.update(exclusive_names)
        clean: list[Any] = []
        for label in labels:
            name = str(getattr(label, "name", label))
            if name in names_to_drop:
                continue
            if label not in clean:
                clean.append(label)
        return clean or [top]
