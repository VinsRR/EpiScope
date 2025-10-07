"""
providers/gemini.py

Google Gemini LLM provider implementation.  Requires the
``google-generativeai`` package and an API key set in
``settings.ProviderConfig.api_key``.  At the time of writing this module
is a stub and demonstrates how to integrate another proprietary LLM
backend.  Note that usage may be subject to rate limits and cost.
"""
from __future__ import annotations

from typing import Iterable

try:
    import google.generativeai as genai  # type: ignore
except ImportError as e:
    genai = None

from ..core.provider import AbstractProvider, ProviderFactory
from ..settings import CONFIG


class GeminiProvider(AbstractProvider):
    """Provider that calls Google's Gemini generative AI API."""

    def __init__(self, model: str = "gemini-pro", api_key: str | None = None) -> None:
        if genai is None:
            raise ImportError(
                "google-generativeai is required for the Gemini provider; install it via `pip install google-generativeai`."
            )
        self.model = model
        self.api_key = api_key or CONFIG.provider.api_key
        if not self.api_key:
            raise ValueError("GeminiProvider requires an API key; set EPISCOPE_PROVIDER_API_KEY")
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(model)

    def chat(self, messages: Iterable[dict[str, str]], **kwargs: str) -> str:
        # For Gemini, we convert messages into a single prompt.  Only the
        # user's messages are concatenated; system prompts are ignored.
        user_parts = [m["content"] for m in messages if m.get("role") != "system"]
        prompt = "\n".join(user_parts)
        response = self._client.generate_content(prompt)
        return response.candidates[0].text.strip()  # type: ignore[index]


ProviderFactory.register_provider("gemini", GeminiProvider)

