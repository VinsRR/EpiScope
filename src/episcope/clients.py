"""
LLM clients for interfacing with different providers (Ollama, OpenAI, etc.).

This module provides a provider-agnostic interface `LLMClient` and concrete
implementations for various services. This allows generators and other
components to be written without being tied to a specific LLM provider.
"""

from __future__ import annotations

import hashlib
import logging
import os
import json
import inspect
import re
from typing import Any, Dict, List, Sequence, Protocol, Optional, Mapping, Tuple
from dataclasses import dataclass, field

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

anthropic: Any
try:
    import anthropic
except ImportError:  # pragma: no cover - exercised in environments without the SDK
    anthropic = None


# Provider-agnostic interface


class LLMClient(Protocol):
    """Minimal interface to support multiple LLM providers."""

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for a chat-style prompt.

        ``response_schema`` (a Pydantic model class or JSON-schema mapping), when
        provided, is translated to the provider's native structured-output
        mechanism so the returned text is constrained to the schema. Providers
        that cannot honor it degrade gracefully to unconstrained decoding.
        """

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


# ---------------------------------------------------------------------------
# Prompt-cache helpers
# ---------------------------------------------------------------------------

def _is_anthropic_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("anthropic/") or "claude" in m


def _with_anthropic_cache_control(
    messages: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Wrap the last system message's content in an Anthropic cache_control block.

    For non-system messages, and for providers that auto-cache, this is a no-op
    (those messages are returned unchanged).
    """
    result: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            result.append({"role": "system", "content": content})
        else:
            result.append(dict(msg))
    return result


# ---------------------------------------------------------------------------
# Structured-output helpers
# ---------------------------------------------------------------------------

# A ``response_schema`` may be a Pydantic model class or an already-built JSON
# schema mapping. Providers each translate it to their own native mechanism.
ResponseSchema = Any


def _normalize_response_schema(response_schema: ResponseSchema) -> Optional[Dict[str, Any]]:
    """Coerce a Pydantic model class or JSON-schema mapping to a plain dict.

    Returns ``None`` when no schema is supplied so callers can treat the
    "unconstrained" case uniformly.
    """
    if response_schema is None:
        return None
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        return response_schema.model_json_schema()
    if isinstance(response_schema, Mapping):
        return dict(response_schema)
    raise TypeError(
        "response_schema must be a Pydantic model class or a mapping, "
        f"got {type(response_schema)!r}"
    )


def _schema_name(schema: Mapping[str, Any], default: str = "response") -> str:
    """A safe identifier for provider APIs that name the schema/tool."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(schema.get("title") or default))[:64]
    return name or default


# JSON Schema keywords that provider "strict" structured-output modes reject.
_STRICT_UNSUPPORTED_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "default",
        "title",
        "minProperties",
        "maxProperties",
    }
)


def _is_open_map(node: Any) -> bool:
    """True for an open-ended object such as ``Dict[str, float]``."""
    return (
        isinstance(node, dict)
        and node.get("type") == "object"
        and "properties" not in node
        and isinstance(node.get("additionalProperties"), dict)
    )


def _is_strict_unrepresentable(node: Any) -> bool:
    """True if a property cannot be expressed under strict mode (open maps)."""
    if not isinstance(node, dict):
        return False
    if _is_open_map(node):
        return True
    branches = node.get("anyOf")
    if isinstance(branches, list):
        return any(_is_strict_unrepresentable(b) for b in branches)
    return False


def _strictify_json_schema(schema: Any) -> Any:
    """Rewrite a JSON schema to satisfy OpenAI/Gemini strict structured output.

    On every object it sets ``additionalProperties: false`` and marks all
    properties ``required`` (optional fields stay nullable via their existing
    ``anyOf``/null union), strips unsupported validation keywords, and drops
    properties that are open maps (e.g. ``class_probabilities``), which strict
    mode cannot represent. These are decode-time constraints only; the full
    Pydantic model still validates the returned JSON, so dropped/optional
    fields simply come back absent.
    """
    if isinstance(schema, list):
        return [_strictify_json_schema(node) for node in schema]
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in _STRICT_UNSUPPORTED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {
                prop: _strictify_json_schema(sub)
                for prop, sub in value.items()
                if not _is_strict_unrepresentable(sub)
            }
        else:
            out[key] = _strictify_json_schema(value)

    if out.get("type") == "object" or "properties" in out:
        props = out.setdefault("properties", {})
        out["additionalProperties"] = False
        out["required"] = list(props.keys())
    return out


def _openai_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI/OpenRouter Structured Outputs ``response_format`` block."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _schema_name(schema),
            "schema": _strictify_json_schema(schema),
            "strict": True,
        },
    }


def _openai_create_with_fallback(create_fn: Any, create_kwargs: Dict[str, Any]) -> Any:
    """Call an OpenAI-style ``create``, retrying unconstrained if the model or
    route rejects ``response_format`` (best-effort structured output)."""
    try:
        return create_fn(**create_kwargs)
    except Exception as exc:  # noqa: BLE001 - provider errors vary by SDK/route
        if "response_format" in create_kwargs:
            logger.warning(
                "Structured output rejected by provider (%s); retrying unconstrained.",
                exc,
            )
            create_kwargs.pop("response_format", None)
            return create_fn(**create_kwargs)
        raise


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
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
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

        # Structured output: Ollama accepts a full JSON schema as `format`,
        # which llama.cpp compiles to a GBNF grammar for constrained decoding.
        # A schema takes precedence over any `format="json"` passed via kwargs.
        schema = _normalize_response_schema(response_schema)
        if schema is not None:
            payload["format"] = schema

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
    - `prompt_cache`: when True, injects Anthropic cache_control blocks for
      Anthropic models (detected by model name). Non-Anthropic models routed
      through OpenRouter use automatic prefix caching — no extra markers needed.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    site_url: str = "http://localhost:8501"  # Default for local Streamlit
    app_title: str = "EpiScope"
    prompt_cache: bool = True
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
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter Chat Completions endpoint."""
        processed = (
            _with_anthropic_cache_control(messages)
            if self.prompt_cache and _is_anthropic_model(model)
            else list(messages)
        )
        sig = inspect.signature(self._client.chat.completions.create)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        create_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=processed,
            temperature=temperature,
            max_tokens=max_tokens,
            **supported_kwargs,
        )
        schema = _normalize_response_schema(response_schema)
        if schema is not None:
            create_kwargs["response_format"] = _openai_response_format(schema)
        response = _openai_create_with_fallback(
            self._client.chat.completions.create, create_kwargs
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
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenAI Chat Completions endpoint."""
        sig = inspect.signature(self._client.chat.completions.create)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        create_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **supported_kwargs,
        )
        schema = _normalize_response_schema(response_schema)
        if schema is not None:
            create_kwargs["response_format"] = _openai_response_format(schema)
        response = _openai_create_with_fallback(
            self._client.chat.completions.create, create_kwargs
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
    - `prompt_cache`: when True, creates an explicit CachedContent for the
      system instruction and reuses it for all calls sharing the same system
      prompt + model combination (keyed by a hash of the system text). Falls
      back silently to uncached if the system prompt is too short for the
      model's minimum cache token threshold.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    prompt_cache: bool = True
    _client: Any = field(init=False, repr=False)
    _cache_store: Dict[Tuple[str, str], str] = field(init=False, repr=False)

    def __post_init__(self):
        self._init_usage_tracking()
        self._cache_store = {}
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

    def _get_or_create_cache(self, model: str, system_text: str) -> Optional[str]:
        """Return a CachedContent name for system_text, creating it on first call.

        Returns None if creation fails (e.g. prompt is below the model's minimum
        token threshold), so the caller can fall back to uncached mode.
        """
        key = (model, hashlib.sha256(system_text.encode()).hexdigest()[:16])
        if key in self._cache_store:
            return self._cache_store[key]
        try:
            cache = self._client.caches.create(
                model=model,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=system_text,
                    ttl="3600s",
                ),
            )
            self._cache_store[key] = cache.name
            return cache.name
        except Exception:
            return None

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API."""
        system_text: Optional[str] = None
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                )
            else:
                text = str(content)
            if role == "system":
                system_text = text
            else:
                gemini_role = "model" if role == "assistant" else role
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        cache_name: Optional[str] = None
        if self.prompt_cache and system_text:
            cache_name = self._get_or_create_cache(model, system_text)

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        else:
            config_kwargs["system_instruction"] = system_text

        # Structured output: Gemini controlled generation via a standard JSON
        # schema (google-genai's `response_json_schema`) lives inside the config
        # object, not among generate_content's kwargs.
        schema = _normalize_response_schema(response_schema)
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = _strictify_json_schema(schema)

        sig = inspect.signature(self._client.models.generate_content)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        def _generate(cfg: Dict[str, Any]) -> Any:
            return self._client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**cfg),
                **supported_kwargs,
            )

        try:
            response = _generate(config_kwargs)
        except Exception as exc:  # noqa: BLE001 - provider errors vary by SDK
            if "response_json_schema" in config_kwargs:
                logger.warning(
                    "Structured output rejected by Gemini (%s); retrying unconstrained.",
                    exc,
                )
                config_kwargs.pop("response_json_schema", None)
                config_kwargs.pop("response_mime_type", None)
                response = _generate(config_kwargs)
            else:
                raise
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


@dataclass
class AnthropicClient(UsageTrackingMixin, LLMClient):
    """
    Client for Anthropic's Claude models via the anthropic SDK.

    - Reads `ANTHROPIC_API_KEY` from the environment.
    - `api_key` can be passed explicitly to override env var.
    - `prompt_cache`: when True, marks the system prompt with an ephemeral
      cache_control block so repeated calls sharing the same system prompt
      and model reuse Anthropic's server-side prompt cache.
    - `default_max_tokens`: Anthropic's Messages API requires `max_tokens`;
      this is used whenever the caller does not supply one.

    Example models: 'claude-opus-4-8', 'claude-sonnet-5'
    """

    api_key: Optional[str] = field(default=None, repr=False)
    prompt_cache: bool = True
    default_max_tokens: int = 4096
    _client: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._init_usage_tracking()
        if anthropic is None:
            raise ImportError(
                "The anthropic package is not installed. Install it to use AnthropicClient."
            )
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "`api_key` not provided and `ANTHROPIC_API_KEY` env var not set."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: ResponseSchema = None,
        **kwargs: Any,
    ) -> str:
        """Call the Anthropic Messages API.

        System messages are pulled out of `messages` into the top-level
        `system` parameter, as required by the Messages API.

        Anthropic has no JSON-schema response format; its native mechanism for
        structured output is forced tool use, so ``response_schema`` is exposed
        as a single tool the model is compelled to call, and that tool's input
        is returned as the JSON string.
        """
        system_text: Optional[str] = None
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text = (
                content
                if isinstance(content, str)
                else "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            )
            if role == "system":
                system_text = f"{system_text}\n\n{text}" if system_text else text
            else:
                converted.append({"role": role, "content": text})

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if system_text:
            create_kwargs["system"] = (
                [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if self.prompt_cache
                else system_text
            )

        sig = inspect.signature(self._client.messages.create)
        create_kwargs.update(
            {k: v for k, v in kwargs.items() if k in sig.parameters}
        )

        # Structured output via forced tool use. Anthropic tool `input_schema`
        # accepts full JSON Schema (open maps, optional fields), so pass the raw
        # schema rather than the strict-mode rewrite.
        schema = _normalize_response_schema(response_schema)
        if schema is not None:
            tool_name = _schema_name(schema)
            create_kwargs["tools"] = [
                {
                    "name": tool_name,
                    "description": "Return the result as structured JSON.",
                    "input_schema": schema,
                }
            ]
            create_kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        try:
            response = self._client.messages.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - provider errors vary by SDK
            if "tools" in create_kwargs:
                logger.warning(
                    "Structured output rejected by Anthropic (%s); retrying unconstrained.",
                    exc,
                )
                create_kwargs.pop("tools", None)
                create_kwargs.pop("tool_choice", None)
                response = self._client.messages.create(**create_kwargs)
            else:
                raise

        usage = getattr(response, "usage", None)
        self._record_usage(
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            cached_tokens=getattr(usage, "cache_read_input_tokens", None),
        )
        tool_inputs = [
            block.input
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        if tool_inputs:
            return json.dumps(tool_inputs[0])
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def embed(
        self,
        texts: List[str],
        *,
        model: str,
        **kwargs: Any,
    ) -> List[List[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Use a different "
            "embedding provider (huggingface, openai, gemini, or ollama)."
        )
