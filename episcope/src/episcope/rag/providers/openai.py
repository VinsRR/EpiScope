"""
providers/openai.py

OpenAI LLM provider implementation.  Requires the ``openai`` package and
an API key specified in ``settings.ProviderConfig.api_key``.  This
provider currently supports simple chat completions via the
``v1/chat/completions`` endpoint.
"""
from __future__ import annotations

from typing import Iterable

import openai

from ..core.provider import AbstractProvider, ProviderFactory
from ..settings import CONFIG


class OpenAIProvider(AbstractProvider):
    """Provider that calls the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self.model = model
        # Use provided key or fall back to global config
        self.api_key = api_key or CONFIG.provider.api_key
        if not self.api_key:
            raise ValueError("OpenAIProvider requires an API key; set EPISCOPE_PROVIDER_API_KEY")
        openai.api_key = self.api_key

    def chat(self, messages: Iterable[dict[str, str]], **kwargs: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=list(messages),
            **kwargs,
        )
        return response["choices"][0]["message"]["content"].strip()


ProviderFactory.register_provider("openai", OpenAIProvider)

