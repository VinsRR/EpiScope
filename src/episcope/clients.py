"""
LLM clients for interfacing with different providers (Ollama, OpenAI, etc.).

This module provides a provider-agnostic interface `LLMClient` and concrete
implementations for various services. This allows generators and other
components to be written without being tied to a specific LLM provider.
"""

from __future__ import annotations

import os
import json
import inspect
from typing import Any, Dict, List, Sequence, Protocol, Optional, Mapping
from dataclasses import dataclass, field

import requests

OpenAI: Any
try:
    from openai import OpenAI as OpenAI
except ImportError:  # pragma: no cover - exercised in environments without the SDK
    OpenAI = None

genai: Any
genai_types: Any
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised in environments without the SDK
    genai = None
    genai_types = None


# Provider-agnostic interface


class LLMClient(Protocol):
    """Minimal interface to support multiple LLM providers."""

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for a chat-style prompt."""

    def embed(
        self,
        texts: List[str],
        *,
        model: str,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Return embeddings for a list of texts."""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    call_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "call_count": self.call_count,
        }

    def copy(self) -> "TokenUsage":
        return TokenUsage(**self.to_dict())


class UsageTrackingMixin:
    """Small helper for providers that can surface token usage."""

    def _init_usage_tracking(self) -> None:
        self.last_usage = TokenUsage()
        self.cumulative_usage = TokenUsage()

    def usage_snapshot(self) -> TokenUsage:
        return self.cumulative_usage.copy()

    def _record_usage(
        self,
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        increment_calls: bool = True,
    ) -> None:
        usage = TokenUsage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
            cached_tokens=int(cached_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
            call_count=1 if increment_calls else 0,
        )
        if usage.total_tokens == 0:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        self.last_usage = usage
        self.cumulative_usage.prompt_tokens += usage.prompt_tokens
        self.cumulative_usage.completion_tokens += usage.completion_tokens
        self.cumulative_usage.total_tokens += usage.total_tokens
        self.cumulative_usage.cached_tokens += usage.cached_tokens
        self.cumulative_usage.reasoning_tokens += usage.reasoning_tokens
        self.cumulative_usage.call_count += usage.call_count


# Provider-specific implementations


@dataclass
class OllamaClient(UsageTrackingMixin, LLMClient):
    """
    Lightweight client for a local Ollama server.

    Reads `OLLAMA_HOST` environment variable, defaulting to http://localhost:11434.
    Uses the /api/chat endpoint; most recent Ollama supports chat roles.

    Example models: 'llama3.1', 'qwen2.5:14b', 'mistral:instruct'
    """

    base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    timeout_s: int = 600
    # If you prefer non-streaming responses; we stitch streamed chunks anyway.
    stream: bool = True

    def __post_init__(self):
        self._init_usage_tracking()

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        # API payload structure; we'll filter kwargs into this.
        # See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
        SUPPORTED_TOP_LEVEL_KWARGS = {"format", "keep_alive", "template"}
        SUPPORTED_OPTIONS_KWARGS = {
            "mirostat",
            "mirostat_eta",
            "mirostat_tau",
            "num_ctx",
            "num_gqa",
            "num_gpu",
            "num_thread",
            "repeat_last_n",
            "repeat_penalty",
            "seed",
            "stop",
            "tfs_z",
            "top_k",
            "top_p",
        }

        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "options": {"temperature": temperature},
            "stream": self.stream,
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # Filter and apply supported kwargs
        for key, value in kwargs.items():
            if key in SUPPORTED_TOP_LEVEL_KWARGS:
                payload[key] = value
            elif key in SUPPORTED_OPTIONS_KWARGS:
                payload["options"][key] = value

        # POST and handle (possibly streaming) response
        with requests.post(
            url, json=payload, timeout=self.timeout_s, stream=self.stream
        ) as r:
            r.raise_for_status()
            if not self.stream:
                data = r.json()
                self._record_usage(
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                )
                return data.get("message", {}).get("content", "")

            parts: List[str] = []
            final_obj: Dict[str, Any] | None = None
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # Each line is a JSON object
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message", {})
                content = msg.get("content")
                if content:
                    parts.append(content)
                if obj.get("done"):
                    final_obj = obj
                    break
            if final_obj is not None:
                self._record_usage(
                    prompt_tokens=final_obj.get("prompt_eval_count"),
                    completion_tokens=final_obj.get("eval_count"),
                )
            return "".join(parts)

    def embed(
        self,
        texts: List[str],
        *,
        model: str,
        **kwargs: Any,
    ) -> List[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        embeddings = []
        for text in texts:
            payload = {"model": model, "prompt": text}
            with requests.post(url, json=payload, timeout=self.timeout_s) as r:
                r.raise_for_status()
                response_data = r.json()
                embeddings.append(response_data.get("embedding", []))
        return embeddings


@dataclass
class OpenRouterClient(UsageTrackingMixin, LLMClient):
    """
    Client for the OpenRouter API (OpenAI-compatible).

    - Reads `OPENROUTER_API_KEY` from environment.
    - `api_key` can be passed explicitly to override env var.
    - `site_url` and `app_title` can be passed for analytics headers.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    site_url: str = "http://localhost:8501"  # Default for local Streamlit
    app_title: str = "EpiScope"
    _client: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._init_usage_tracking()
        if OpenAI is None:
            raise ImportError(
                "The openai package is not installed. Install it to use OpenRouterClient."
            )
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "`api_key` not provided and `OPENROUTER_API_KEY` env var not set."
            )

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            default_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_title,
            },
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter Chat Completions endpoint."""
        sig = inspect.signature(self._client.chat.completions.create)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **supported_kwargs,
        )
        usage = getattr(response, "usage", None)
        self._record_usage(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            reasoning_tokens=getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            ),
            cached_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
        )
        return response.choices[0].message.content or ""

    def embed(
        self,
        texts: List[str],
        *,
        model: str,
        **kwargs: Any,
    ) -> List[List[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=model,
            **kwargs,
        )
        return [item.embedding for item in response.data]


# Proprietary


@dataclass
# OpenRouter could even just be called from OpenAIClient since it's compatible. Keeping separate for clarity.
class OpenAIClient(UsageTrackingMixin, LLMClient):
    """
    Client for OpenAI API compatible endpoints (including Azure).

    - Reads `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` from environment.
    - `api_key` and `base_url` can be passed explicitly to override env vars.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = field(default=None)
    _client: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._init_usage_tracking()
        if OpenAI is None:
            raise ImportError(
                "The openai package is not installed. Install it to use OpenAIClient."
            )
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "`api_key` not provided and `OPENAI_API_KEY` env var not set."
            )
        url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=key, base_url=url)

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenAI Chat Completions endpoint."""
        sig = inspect.signature(self._client.chat.completions.create)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **supported_kwargs,
        )
        usage = getattr(response, "usage", None)
        self._record_usage(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            reasoning_tokens=getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            ),
            cached_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
        )
        return response.choices[0].message.content or ""

    def embed(
        self,
        texts: List[str],
        *,
        model: str,
        **kwargs: Any,
    ) -> List[List[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=model,
            **kwargs,
        )
        return [item.embedding for item in response.data]


@dataclass
class GeminiClient(UsageTrackingMixin, LLMClient):
    """
    Client for Google's Gemini models via the google-genai SDK.

    - Reads `GEMINI_API_KEY` from the environment.
    - `api_key` can be passed explicitly to override env var.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    _client: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._init_usage_tracking()
        if genai is None or genai_types is None:
            raise ImportError(
                "The google-genai package is not installed. Install it to use GeminiClient."
            )
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "`api_key` not provided and `GEMINI_API_KEY` env var not set."
            )
        self._client = genai.Client(api_key=key)

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API."""
        full_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        generation_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        sig = inspect.signature(self._client.models.generate_content)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        response = self._client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=generation_config,
            **supported_kwargs,
        )
        usage = getattr(response, "usage_metadata", None)
        self._record_usage(
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            cached_tokens=getattr(usage, "cached_content_token_count", None),
            reasoning_tokens=getattr(usage, "thoughts_token_count", None),
        )
        return response.text or ""

    def embed(
        self,
        texts: List[str],
        *,
        model: str = "gemini-embedding-001",
        output_dimensionality: Optional[int] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        **kwargs: Any,
    ) -> List[List[float]]:
        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        result = self._client.models.embed_content(
            model=model,
            contents=texts,
            config=config,
            **kwargs,
        )
        return [embedding.values or [] for embedding in result.embeddings]
