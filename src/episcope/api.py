from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from episcope.clients import GeminiClient, LLMClient, OllamaClient, OpenAIClient, OpenRouterClient
from episcope.db.mongo_academic_db import MongoAcademicDB
from episcope.rag.generation.llm_generator import LLMGenerator
from episcope.rag.retrieval.candidates import (
    HybridCandidateRetriever,
    SemanticCandidateRetriever,
    SparseCandidateRetriever,
)
from episcope.rag.retrieval.retriever import Retriever
from episcope.settings import AppSettings, env
from episcope.vectordb.qdrant import QdrantDB
from episcope.workflows import PaperClassifier, PrecisionMiner
from episcope.workflows.classification import (
    DataAccessibilityClassifierConfig,
    DataTypeClassifierConfig,
    GeoClassifierConfig,
    GlobalCrossEncoderReranker,
    PaperTypeClassifierConfig,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.precision_miner import (
    FindDataSourcesConfig,
    FindSupplementaryLinksConfig,
    IdentifyKeyReferencesConfig,
)

app = FastAPI(title="EpiScope API", version="0.2.0")


def _settings() -> AppSettings:
    return AppSettings.from_env()


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {str(key): _json_ready(item) for key, item in value.model_dump().items()}
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


class BackendConfig(BaseModel):
    strategy_name: str = Field(default_factory=lambda: _settings().strategy_name)
    mongo_uri: Optional[str] = Field(default_factory=lambda: _settings().mongo_uri)
    mongo_db_name: str = Field(default_factory=lambda: _settings().mongo_db_name)
    qdrant_url: str = Field(default_factory=lambda: _settings().qdrant_url)
    qdrant_collection: str = Field(default_factory=lambda: _settings().qdrant_collection)
    llm_provider: Literal["gemini", "openai", "openrouter", "ollama"] = Field(
        default_factory=lambda: _settings().llm_provider
    )
    llm_model: str = Field(default_factory=lambda: _settings().llm_model)
    llm_temperature: float = 0.0
    workflow_top_k: int = 10
    retrieval_mode: Literal["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"] = "hybrid"
    evidence_reranker_kind: Literal["none", "global_cross_encoder", "within_label_cross_encoder"] = "none"
    cross_encoder_model: Optional[str] = Field(default_factory=lambda: _settings().cross_encoder_model)
    cross_encoder_top_k: Optional[int] = 15


class ClassificationRequest(BaseModel):
    paper_id: str
    classifier_kind: Literal["paper_type", "data_accessibility", "data_type", "geo"] = "data_accessibility"
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)


class PrecisionMinerRequest(BaseModel):
    paper_id: str
    miner_kind: Literal["find_data_sources", "find_supplementary_links", "identify_key_references"] = "find_data_sources"
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)


class ExplorerRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    generate_answer: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)
    config: BackendConfig = Field(default_factory=BackendConfig)


def _build_llm_client(provider: str) -> LLMClient:
    provider = provider.lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "openrouter":
        return OpenRouterClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unsupported llm_provider={provider!r}")


def _build_retriever(config: BackendConfig) -> Retriever:
    try:
        vdb = QdrantDB(collection=config.qdrant_collection, url=config.qdrant_url)
    except Exception as exc:
        message = str(exc)
        if "dense_dim must be provided" in message:
            raise ValueError(
                f"Qdrant collection {config.qdrant_collection!r} was not found at {config.qdrant_url}. "
                "The UI expects an existing indexed collection."
            ) from exc
        raise
    if config.retrieval_mode == "dense_only":
        return Retriever(
            vectordb=vdb,
            candidate_retrievers=[SemanticCandidateRetriever(vdb)],
            use_rerank=False,
        )
    if config.retrieval_mode == "sparse_only":
        return Retriever(
            vectordb=vdb,
            candidate_retrievers=[SparseCandidateRetriever(vdb)],
            use_rerank=False,
        )
    if config.retrieval_mode == "hybrid_candidates_only":
        return Retriever(
            vectordb=vdb,
            candidate_retrievers=[HybridCandidateRetriever(vdb)],
            use_rerank=False,
        )
    if config.retrieval_mode == "hybrid":
        return Retriever(vectordb=vdb, use_rerank=False)
    raise ValueError(f"Unsupported retrieval_mode={config.retrieval_mode!r}")


def _build_generator(config: BackendConfig) -> LLMGenerator:
    client = _build_llm_client(config.llm_provider)
    return LLMGenerator(
        client=client,
        model=config.llm_model,
        temperature=config.llm_temperature,
    )


def _build_db(config: BackendConfig) -> MongoAcademicDB:
    if not config.mongo_uri:
        raise ValueError(
            "No Mongo URI configured. Set config.mongo_uri or provide the MONGO_URI environment variable."
        )
    return MongoAcademicDB(uri=config.mongo_uri, db_name=config.mongo_db_name)


def _health_checks(defaults: BackendConfig) -> Dict[str, Any]:
    return {
        "mongo_uri_configured": bool(defaults.mongo_uri),
        "qdrant_url": defaults.qdrant_url,
        "qdrant_collection": defaults.qdrant_collection,
        "llm_provider": defaults.llm_provider,
        "llm_api_key_configured": bool(
            {
                "gemini": env("GEMINI_API_KEY"),
                "openai": env("OPENAI_API_KEY"),
                "openrouter": env("OPENROUTER_API_KEY"),
                "ollama": env("OLLAMA_HOST", "http://localhost:11434"),
            }.get(defaults.llm_provider)
        ),
    }


def _classifier_config(kind: str, top_k: int):
    config_map = {
        "paper_type": PaperTypeClassifierConfig,
        "data_accessibility": DataAccessibilityClassifierConfig,
        "data_type": DataTypeClassifierConfig,
        "geo": GeoClassifierConfig,
    }
    config = config_map[kind]()
    config.top_k = top_k
    return config


def _precision_miner_config(kind: str, top_k: int):
    config_map = {
        "find_data_sources": FindDataSourcesConfig,
        "find_supplementary_links": FindSupplementaryLinksConfig,
        "identify_key_references": IdentifyKeyReferencesConfig,
    }
    config = config_map[kind]()
    config.top_k = top_k
    return config


def _evidence_reranker(config: BackendConfig):
    if config.evidence_reranker_kind == "none":
        return None
    if not config.cross_encoder_model:
        raise ValueError("cross_encoder_model must be set when evidence_reranker_kind is not 'none'.")
    if config.evidence_reranker_kind == "global_cross_encoder":
        return GlobalCrossEncoderReranker.from_huggingface(
            model_name=config.cross_encoder_model,
            top_k=config.cross_encoder_top_k,
        )
    if config.evidence_reranker_kind == "within_label_cross_encoder":
        return WithinLabelCrossEncoderReranker.from_huggingface(
            model_name=config.cross_encoder_model,
            top_k=config.cross_encoder_top_k,
        )
    raise ValueError(f"Unsupported evidence_reranker_kind={config.evidence_reranker_kind!r}")


@app.get("/health")
def health() -> Dict[str, Any]:
    defaults = BackendConfig()
    return {
        "status": "ok",
        "service": "episcope-api",
        "defaults": _json_ready(defaults),
        "checks": _health_checks(defaults),
    }


@app.post("/classify")
def classify(request: ClassificationRequest) -> Dict[str, Any]:
    try:
        db = _build_db(request.config)
        classifier = PaperClassifier(
            retriever=_build_retriever(request.config),
            generator=_build_generator(request.config),
            strategy_name=request.config.strategy_name,
            academic_db=db,
            config=_classifier_config(request.classifier_kind, request.config.workflow_top_k),
            evidence_reranker=_evidence_reranker(request.config),
        )
        if request.detailed:
            result = classifier.run_detailed(request.paper_id)
        else:
            result = classifier.run(request.paper_id)
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc


@app.post("/precision-miner")
def precision_miner(request: PrecisionMinerRequest) -> Dict[str, Any]:
    try:
        db = _build_db(request.config)
        miner = PrecisionMiner(
            retriever=_build_retriever(request.config),
            generator=_build_generator(request.config),
            strategy_name=request.config.strategy_name,
            academic_db=db,
            config=_precision_miner_config(request.miner_kind, request.config.workflow_top_k),
        )
        if request.detailed:
            result = miner.run_detailed(request.paper_id)
        else:
            result = miner.run(request.paper_id)
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Precision miner failed: {exc}") from exc


@app.post("/explore")
def explore(request: ExplorerRequest) -> Dict[str, Any]:
    try:
        retriever = _build_retriever(request.config)
        retrieved = list(
            retriever.retrieve(
                request.query,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                filter=request.filters or None,
            )
        )

        response: Dict[str, Any] = {
            "query": request.query,
            "top_k": request.top_k,
            "similarity_threshold": request.similarity_threshold,
            "generate_answer": request.generate_answer,
            "filters": request.filters,
            "retrieved_chunks": [_json_ready(chunk) for chunk in retrieved],
            "retrieval_count": len(retrieved),
            "answer": None,
            "provenance": None,
        }

        if request.generate_answer:
            generator = _build_generator(request.config)
            provenance = generator.generate(retrieved, question=request.query)
            response["answer"] = provenance.answer
            response["provenance"] = _json_ready(provenance)

        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explorer failed: {exc}") from exc
