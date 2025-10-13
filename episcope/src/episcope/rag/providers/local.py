"""
providers/local.py

Local LLM provider implementation using Ollama.  This provider uses the
``ollama`` Python package to send chat messages to a locally running
Ollama instance and return the generated response.
"""
from __future__ import annotations

from typing import Iterable, Optional

import ollama

from ..core.provider import AbstractProvider, ProviderFactory


class LocalProvider(AbstractProvider):
    """Provider that uses a locally running Ollama instance."""

    def __init__(self, model: str = "llama3:8b", **_: Optional[str]) -> None:
        self.model = model

    def chat(self, messages: Iterable[dict[str, str]], **kwargs: str) -> str:
        response = ollama.chat(model=self.model, messages=list(messages))
        # ``response`` can be a dataclass or dict; convert as needed
        if hasattr(response, "message"):
            return response.message.content.strip()
        return response.get("message", {}).get("content", "").strip()


# Register provider on import
ProviderFactory.register_provider("local", LocalProvider)

