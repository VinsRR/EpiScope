import logging
import os
from typing import Optional

from .base import Embedder
from .huggingface import (
    HuggingFaceEmbedder,
    HuggingFaceSparseEmbedder,
    HuggingFaceLateEmbedder,
)
from .ollama import OllamaEmbedder
from .openai import OpenAIEmbedder
from .gemini import GeminiEmbedder

logger = logging.getLogger(__name__)

# Known OpenAI embedding model identifiers (used only by auto-detection).
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
}

# Provider name -> dense embedder class. Adding a provider is one entry here.
EMBEDDER_PROVIDERS = {
    "huggingface": HuggingFaceEmbedder,
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "ollama": OllamaEmbedder,
}


class EmbedderFactory:
    """Instantiates the correct dense embedder for a model.

    Selection is provider-driven, mirroring the LLM client side:

    1. an explicit ``provider`` argument, else
    2. the ``EPISCOPE_EMBED_PROVIDER`` environment variable, else
    3. ``"auto"`` — infer the provider from the model name for common,
       unambiguous cases.

    When the provider cannot be determined this raises a ``ValueError`` — it
    never silently falls back to a single provider.
    """

    @staticmethod
    def get_embedder(
        model_name: str,
        *,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Embedder:
        resolved = (
            (provider or os.environ.get("EPISCOPE_EMBED_PROVIDER") or "auto")
            .strip()
            .lower()
        )
        if resolved == "auto":
            resolved = EmbedderFactory._detect_provider(model_name)

        embedder_cls = EMBEDDER_PROVIDERS.get(resolved)
        if embedder_cls is None:
            raise ValueError(
                f"Unknown embedding provider {resolved!r}. Choose one of: "
                f"{', '.join(sorted(EMBEDDER_PROVIDERS))} (or 'auto')."
            )
        logger.info(
            "Creating %s for model '%s' (provider=%s).",
            embedder_cls.__name__,
            model_name,
            resolved,
        )
        return embedder_cls(model=model_name, **kwargs)

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        """Infer a provider from the model name for common cases, else raise."""
        name = model_name.strip()
        lowered = name.lower()
        if "/" in name and not name.startswith("models/"):
            return "huggingface"
        if name.startswith("models/") or "gemini" in lowered:
            return "gemini"
        if (
            name in OPENAI_EMBEDDING_MODELS
            or name.startswith("text-embedding-3")
            or "ada-002" in lowered
        ):
            return "openai"
        raise ValueError(
            f"Could not infer an embedding provider from model name {model_name!r}. "
            f"Pass provider=... or set EPISCOPE_EMBED_PROVIDER to one of: "
            f"{', '.join(sorted(EMBEDDER_PROVIDERS))}."
        )

    @staticmethod
    def get_sparse_embedder(model_name: str, **kwargs) -> HuggingFaceSparseEmbedder:
        """Factory method to get a sparse embedder instance."""
        logger.info(f"Creating HuggingFaceSparseEmbedder for model '{model_name}'.")
        return HuggingFaceSparseEmbedder(model=model_name, **kwargs)

    @staticmethod
    def get_late_embedder(model_name: str, **kwargs) -> HuggingFaceLateEmbedder:
        """Factory method to get a late-interaction embedder instance."""
        logger.info(f"Creating HuggingFaceLateEmbedder for model '{model_name}'.")
        return HuggingFaceLateEmbedder(model=model_name, **kwargs)
