"""
settings.py

Global configuration for the EpiScope project.  This module defines
Pydantic models that capture configuration for the index, retrieval and
LLM provider subsystems.  Settings may be loaded from environment
variables, a .env file, or overridden at runtime.
"""
from __future__ import annotations

from typing import Optional
# Import BaseSettings from pydantic.  In Pydantic v2 the class has
# moved to the separate ``pydantic_settings`` package.  Attempt
# importing from the old location first for compatibility.  If
# pydantic is not installed at all (as may be the case in minimal
# test environments), fall back to a simple dummy implementation
# based on standard dataclasses.
try:
    from pydantic import BaseSettings, Field, validator  # type: ignore[attr-defined]
except ImportError:
    try:
        from pydantic_settings import BaseSettings  # type: ignore
        from pydantic import Field  # type: ignore
        # ``validator`` is unused but imported for backward compatibility
        def validator(*args, **kwargs):  # type: ignore
            def _wrapper(fn):  # pragma: no cover
                return fn
            return _wrapper
    except Exception:
        # Define a minimal BaseSettings fallback
        from dataclasses import dataclass

        def Field(default=None, description: str = ""):  # type: ignore
            return default

        def validator(*args, **kwargs):  # type: ignore
            def _wrapper(fn):  # pragma: no cover
                return fn
            return _wrapper

        @dataclass
        class BaseSettings:
            """Minimal drop‑in replacement for pydantic BaseSettings.

            This fallback simply stores attributes and ignores
            environment variables or validation.  It is sufficient
            for test environments where pydantic is unavailable.
            """
            pass


class ProviderConfig(BaseSettings):
    """Configuration for the large language model provider.

    The ``provider`` field controls which backend to use.  Supported
    providers include ``"local"`` (Ollama/llama.cpp), ``"openai"`` and
    ``"gemini"``.  Additional provider classes may be registered in
    ``providers/factory.py``.
    """

    provider: str = Field(
        "local",
        description="Which LLM provider backend to use. One of: local, openai, gemini.",
    )
    model: str = Field(
        default="llama3:8b",
        description="Name of the model to use with the selected provider.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for proprietary providers. Ignored for local backends.",
    )

    class Config:
        env_prefix = "EPISCOPE_PROVIDER_"


class IndexConfig(BaseSettings):
    """Configuration for the vector index layer.

    These values control the Qdrant collection and embedding dimensions used
    by the retrieval components.  See ``configs/text_only.py`` for
    defaults.
    """

    collection_name: str = Field(
        default="episcope",
        description="Name of the Qdrant collection used for storing embeddings.",
    )
    url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant service.",
    )
    timeout: int = Field(
        default=60,
        description="Timeout (in seconds) for Qdrant requests.",
    )
    distance: str = Field(
        default="COSINE",
        description="Distance metric used for vector similarity. See Qdrant docs.",
    )

    class Config:
        env_prefix = "EPISCOPE_INDEX_"


class AppConfig(BaseSettings):
    """Top-level application configuration.

    Aggregates provider and index settings.  Additional project-wide
    settings may be added here in future.
    """

    provider: ProviderConfig = ProviderConfig()  # type: ignore[assignment]
    index: IndexConfig = IndexConfig()          # type: ignore[assignment]

    class Config:
        env_prefix = "EPISCOPE_"


# Create a singleton configuration object when the module is imported.  This
# object may be imported throughout the codebase to access the current
# settings.  It may be overridden in tests or at runtime by creating a
# new instance of ``AppConfig``.
CONFIG = AppConfig()

