from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import dotenv

dotenv.load_dotenv()


def env(
    name: str, default: Optional[str] = None, *, legacy_names: tuple[str, ...] = ()
) -> Optional[str]:
    for candidate in (name, *legacy_names):
        value = os.getenv(candidate)
        if value is not None and value != "":
            return value
    return default


@dataclass(frozen=True)
class AppSettings:
    strategy_name: str = "grobid"
    mongo_uri: Optional[str] = None
    mongo_db_name: str = "episcope_academic_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope_academic"
    grobid_url: str = "http://localhost:8070"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    embed_provider: str = "auto"
    cross_encoder_model: Optional[str] = None
    ollama_host: str = "http://localhost:11434"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppSettings":
        port_raw = env("EPISCOPE_API_PORT", "8000")
        return cls(
            strategy_name=env("EPISCOPE_STRATEGY_NAME", "grobid") or "grobid",
            mongo_uri=env("MONGO_URI", legacy_names=("mongo_uri",)),
            mongo_db_name=env("MONGO_DB_NAME", legacy_names=("mongo_db_name",))
            or "episcope_academic_db",
            qdrant_url=env("QDRANT_URL", "http://localhost:6333")
            or "http://localhost:6333",
            qdrant_collection=env("QDRANT_COLLECTION", "episcope_academic")
            or "episcope_academic",
            grobid_url=env("GROBID_URL", "http://localhost:8070")
            or "http://localhost:8070",
            llm_provider=(env("EPISCOPE_LLM_PROVIDER", "gemini") or "gemini").lower(),
            llm_model=env("EPISCOPE_LLM_MODEL", "gemini-2.5-flash")
            or "gemini-2.5-flash",
            embed_provider=(env("EPISCOPE_EMBED_PROVIDER", "auto") or "auto").lower(),
            cross_encoder_model=env("CROSS_ENCODER_MODEL"),
            ollama_host=env("OLLAMA_HOST", "http://localhost:11434")
            or "http://localhost:11434",
            api_host=env("EPISCOPE_API_HOST", "0.0.0.0") or "0.0.0.0",
            api_port=int(port_raw) if port_raw else 8000,
            api_base_url=env("EPISCOPE_API_BASE_URL", "http://localhost:8000")
            or "http://localhost:8000",
            log_level=(env("EPISCOPE_LOG_LEVEL", "INFO") or "INFO").upper(),
        )
