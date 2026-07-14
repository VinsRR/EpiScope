from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from episcope.clients import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)
from episcope.db.mongo_academic_db import MongoAcademicDB
from episcope.rag.generation.llm_generator import LLMGenerator
from episcope.rag.provenance import Provenance
from episcope.rag.retrieval.candidates import (
    HybridCandidateRetriever,
    SemanticCandidateRetriever,
    SparseCandidateRetriever,
)
from episcope.rag.retrieval.retriever import Retriever
from episcope.schemas import SearchResult
from episcope.settings import AppSettings, env
from episcope.vectordb.qdrant import QdrantDB
from episcope.workflows import PaperClassifier, PrecisionMiner
from episcope.workflows.classification import (
    GlobalCrossEncoderReranker,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.registry import (
    build_classifier_config as _build_classifier_config,
    build_precision_miner_config as _build_precision_miner_config,
)

LlmProvider = Literal["gemini", "openai", "openrouter", "anthropic", "ollama"]
RetrievalMode = Literal[
    "dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"
]
EvidenceRerankerKind = Literal[
    "none", "global_cross_encoder", "within_label_cross_encoder"
]


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for the corpus-backed EpiScope application runtime."""

    strategy_name: str = "grobid"
    mongo_uri: Optional[str] = None
    mongo_db_name: str = "episcope_academic_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope_academic"
    llm_provider: LlmProvider = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    workflow_top_k: int = 10
    retrieval_mode: RetrievalMode = "hybrid"
    evidence_reranker_kind: EvidenceRerankerKind = "none"
    cross_encoder_model: Optional[str] = None
    cross_encoder_top_k: Optional[int] = 15

    @classmethod
    def from_settings(cls, settings: Optional[AppSettings] = None) -> "RuntimeConfig":
        settings = settings or AppSettings.from_env()
        return cls(
            strategy_name=settings.strategy_name,
            mongo_uri=settings.mongo_uri,
            mongo_db_name=settings.mongo_db_name,
            qdrant_url=settings.qdrant_url,
            qdrant_collection=settings.qdrant_collection,
            llm_provider=_coerce_llm_provider(settings.llm_provider),
            llm_model=settings.llm_model,
            cross_encoder_model=settings.cross_encoder_model,
        )


@dataclass(frozen=True)
class BackendHealth:
    """Runtime defaults and environment checks for UI/API clients."""

    defaults: RuntimeConfig
    checks: Dict[str, Any]


@dataclass(frozen=True)
class ExploreResult:
    query: str
    top_k: int
    similarity_threshold: float
    generate_answer: bool
    filters: Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    retrieval_count: int = 0
    answer: Optional[str] = None
    provenance: Optional[Provenance] = None


class EpiScopeRuntime:
    """Application-facing facade for corpus-backed EpiScope workflows."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig.from_settings()

    def health(self) -> BackendHealth:
        return BackendHealth(defaults=self.config, checks=_health_checks(self.config))

    def explore(
        self,
        query: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        generate_answer: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ExploreResult:
        filters = filters or {}
        retriever = self.build_retriever()
        retrieved = list(
            retriever.retrieve(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filter=filters or None,
            )
        )

        answer = None
        provenance = None
        if generate_answer:
            provenance = self.build_generator().generate(retrieved, question=query)
            answer = provenance.answer

        return ExploreResult(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            generate_answer=generate_answer,
            filters=filters,
            retrieved_chunks=retrieved,
            retrieval_count=len(retrieved),
            answer=answer,
            provenance=provenance,
        )

    def classify(
        self,
        paper_id: str,
        *,
        classifier_kind: str = "data_accessibility",
        config: Optional[Any] = None,
        detailed: bool = True,
    ) -> Any:
        classifier = PaperClassifier(
            retriever=self.build_retriever(),
            generator=self.build_generator(),
            strategy_name=self.config.strategy_name,
            academic_db=self.build_db(),
            config=config or self.build_classifier_config(classifier_kind),
            evidence_reranker=self.build_evidence_reranker(),
        )
        if detailed:
            return classifier.run_detailed(paper_id)
        return classifier.run(paper_id)

    def precision_mine(
        self,
        paper_id: str,
        *,
        miner_kind: str = "find_data_sources",
        config: Optional[Any] = None,
        detailed: bool = True,
    ) -> Any:
        miner = PrecisionMiner(
            retriever=self.build_retriever(),
            generator=self.build_generator(),
            strategy_name=self.config.strategy_name,
            academic_db=self.build_db(),
            config=config or self.build_precision_miner_config(miner_kind),
        )
        if detailed:
            return miner.run_detailed(paper_id)
        return miner.run(paper_id)

    def build_llm_client(self) -> LLMClient:
        provider = self.config.llm_provider.lower()
        if provider == "gemini":
            return GeminiClient()
        if provider == "openai":
            return OpenAIClient()
        if provider == "openrouter":
            return OpenRouterClient()
        if provider == "anthropic":
            return AnthropicClient()
        if provider == "ollama":
            return OllamaClient()
        raise ValueError(f"Unsupported llm_provider={provider!r}")

    def build_generator(self) -> LLMGenerator:
        return LLMGenerator(
            client=self.build_llm_client(),
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
        )

    def build_db(self) -> MongoAcademicDB:
        if not self.config.mongo_uri:
            raise ValueError(
                "No Mongo URI configured. Set config.mongo_uri or provide "
                "the MONGO_URI environment variable."
            )
        return MongoAcademicDB(
            uri=self.config.mongo_uri,
            db_name=self.config.mongo_db_name,
        )

    def build_retriever(self) -> Retriever:
        try:
            vdb = QdrantDB(
                collection=self.config.qdrant_collection,
                url=self.config.qdrant_url,
            )
        except Exception as exc:
            message = str(exc)
            if "dense_dim must be provided" in message:
                raise ValueError(
                    f"Qdrant collection {self.config.qdrant_collection!r} "
                    f"was not found at {self.config.qdrant_url}. "
                    "The application runtime expects an existing indexed collection."
                ) from exc
            raise

        if self.config.retrieval_mode == "dense_only":
            return Retriever(
                vectordb=vdb,
                candidate_retrievers=[SemanticCandidateRetriever(vdb)],
                use_rerank=False,
            )
        if self.config.retrieval_mode == "sparse_only":
            return Retriever(
                vectordb=vdb,
                candidate_retrievers=[SparseCandidateRetriever(vdb)],
                use_rerank=False,
            )
        if self.config.retrieval_mode == "hybrid_candidates_only":
            return Retriever(
                vectordb=vdb,
                candidate_retrievers=[HybridCandidateRetriever(vdb)],
                use_rerank=False,
            )
        if self.config.retrieval_mode == "hybrid":
            return Retriever(vectordb=vdb, use_rerank=False)
        raise ValueError(f"Unsupported retrieval_mode={self.config.retrieval_mode!r}")

    def build_classifier_config(self, kind: str):
        return _build_classifier_config(kind, self.config.workflow_top_k)

    def build_precision_miner_config(self, kind: str):
        return _build_precision_miner_config(kind, self.config.workflow_top_k)

    def build_evidence_reranker(self):
        if self.config.evidence_reranker_kind == "none":
            return None
        if not self.config.cross_encoder_model:
            raise ValueError(
                "cross_encoder_model must be set when "
                "evidence_reranker_kind is not 'none'."
            )
        if self.config.evidence_reranker_kind == "global_cross_encoder":
            return GlobalCrossEncoderReranker.from_huggingface(
                model_name=self.config.cross_encoder_model,
                top_k=self.config.cross_encoder_top_k,
            )
        if self.config.evidence_reranker_kind == "within_label_cross_encoder":
            return WithinLabelCrossEncoderReranker.from_huggingface(
                model_name=self.config.cross_encoder_model,
                top_k=self.config.cross_encoder_top_k,
            )
        raise ValueError(
            f"Unsupported evidence_reranker_kind={self.config.evidence_reranker_kind!r}"
        )


def _health_checks(config: RuntimeConfig) -> Dict[str, Any]:
    return {
        "mongo_uri_configured": bool(config.mongo_uri),
        "qdrant_url": config.qdrant_url,
        "qdrant_collection": config.qdrant_collection,
        "llm_provider": config.llm_provider,
        "llm_api_key_configured": bool(
            {
                "gemini": env("GEMINI_API_KEY"),
                "openai": env("OPENAI_API_KEY"),
                "openrouter": env("OPENROUTER_API_KEY"),
                "anthropic": env("ANTHROPIC_API_KEY"),
                "ollama": env("OLLAMA_HOST", "http://localhost:11434"),
            }.get(config.llm_provider)
        ),
    }


def _coerce_llm_provider(provider: str) -> LlmProvider:
    normalized = provider.lower()
    if normalized in {"gemini", "openai", "openrouter", "anthropic", "ollama"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported llm_provider={provider!r}")
