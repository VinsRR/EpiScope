from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from episcope import __version__
from episcope.services import EpiScopeRuntime, RuntimeConfig
from episcope.services.errors import humanize_error
from episcope.workflows.registry import (
    TaskSpec,
    build_classifier_config_from_spec,
    build_miner_config_from_spec,
    classifier_catalog,
    miner_catalog,
)

app = FastAPI(title="EpiScope API", version=__version__)


def _path_str(value: Optional[Path]) -> Optional[str]:
    return str(value) if value else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {str(key): _json_ready(item) for key, item in value.model_dump().items()}
    if is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


class BackendConfig(BaseModel):
    strategy_name: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().strategy_name
    )
    mongo_uri: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().mongo_uri
    )
    mongo_db_name: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().mongo_db_name
    )
    qdrant_url: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().qdrant_url
    )
    qdrant_collection: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().qdrant_collection
    )
    llm_provider: Literal["gemini", "openai", "openrouter", "anthropic", "ollama"] = (
        Field(default_factory=lambda: RuntimeConfig.from_settings().llm_provider)
    )
    llm_model: str = Field(
        default_factory=lambda: RuntimeConfig.from_settings().llm_model
    )
    llm_temperature: float = 0.0
    workflow_top_k: int = 10
    retrieval_mode: Literal[
        "dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"
    ] = "hybrid"
    evidence_reranker_kind: Literal[
        "none", "global_cross_encoder", "within_label_cross_encoder"
    ] = "none"
    cross_encoder_model: Optional[str] = Field(
        default_factory=lambda: RuntimeConfig.from_settings().cross_encoder_model
    )
    cross_encoder_top_k: Optional[int] = 15
    # Local-first fallback paths, used only when qdrant_url / mongo_uri are
    # unset (see RuntimeConfig). Exposed as str over HTTP; converted back to
    # Path in to_runtime_config().
    index_dir: Optional[str] = Field(
        default_factory=lambda: _path_str(RuntimeConfig.from_settings().index_dir)
    )
    metadata_backup: Optional[str] = Field(
        default_factory=lambda: _path_str(
            RuntimeConfig.from_settings().metadata_backup
        )
    )

    def to_runtime_config(self) -> RuntimeConfig:
        data = self.model_dump()
        data["index_dir"] = Path(data["index_dir"]) if data["index_dir"] else None
        data["metadata_backup"] = (
            Path(data["metadata_backup"]) if data["metadata_backup"] else None
        )
        return RuntimeConfig(**data)


class ClassificationRequest(BaseModel):
    paper_id: str
    classifier_kind: str = "data_accessibility"
    task: Optional[TaskSpec] = None
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)


class PrecisionMinerRequest(BaseModel):
    paper_id: str
    miner_kind: str = "find_data_sources"
    task: Optional[TaskSpec] = None
    detailed: bool = True
    config: BackendConfig = Field(default_factory=BackendConfig)


class ExplorerRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    generate_answer: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)
    config: BackendConfig = Field(default_factory=BackendConfig)


def _redact_secrets(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Never expose connection strings (which may carry credentials) over HTTP.

    ``checks.mongo_uri_configured`` already reports presence as a boolean.
    """
    safe = dict(defaults)
    if safe.get("mongo_uri"):
        safe["mongo_uri"] = "***configured***"
    return safe


@app.get("/health")
def health() -> Dict[str, Any]:
    runtime_health = EpiScopeRuntime().health()
    return {
        "status": "ok",
        "service": "episcope-api",
        "defaults": _redact_secrets(_json_ready(runtime_health.defaults)),
        "checks": runtime_health.checks,
        "kinds": {
            "classifiers": classifier_catalog(),
            "miners": miner_catalog(),
        },
    }


@app.post("/classify")
def classify(request: ClassificationRequest) -> Dict[str, Any]:
    try:
        runtime = EpiScopeRuntime(request.config.to_runtime_config())
        inline_config = None
        if request.task is not None:
            if request.task.kind != "classifier":
                raise ValueError("task.kind must be 'classifier' for /classify.")
            inline_config = build_classifier_config_from_spec(request.task)
        result = runtime.classify(
            request.paper_id,
            classifier_kind=request.classifier_kind,
            config=inline_config,
            detailed=request.detailed,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Classification failed: {humanize_error(exc)}"
        ) from exc


@app.post("/precision-miner")
def precision_miner(request: PrecisionMinerRequest) -> Dict[str, Any]:
    try:
        runtime = EpiScopeRuntime(request.config.to_runtime_config())
        inline_config = None
        if request.task is not None:
            if request.task.kind != "miner":
                raise ValueError("task.kind must be 'miner' for /precision-miner.")
            inline_config = build_miner_config_from_spec(request.task)
        result = runtime.precision_mine(
            request.paper_id,
            miner_kind=request.miner_kind,
            config=inline_config,
            detailed=request.detailed,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Precision miner failed: {humanize_error(exc)}"
        ) from exc


@app.post("/explore")
def explore(request: ExplorerRequest) -> Dict[str, Any]:
    try:
        runtime = EpiScopeRuntime(request.config.to_runtime_config())
        result = runtime.explore(
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            generate_answer=request.generate_answer,
            filters=request.filters,
        )
        return _json_ready(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=humanize_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explorer failed: {humanize_error(exc)}") from exc
