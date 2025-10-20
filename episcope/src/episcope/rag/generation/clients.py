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
from openai import OpenAI
import google.generativeai as genai


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


# Provider-specific implementations

@dataclass
class OllamaClient(LLMClient):
    """
    Lightweight client for a local Ollama server.

    Assumes Ollama is running (default: http://localhost:11434).
    Uses the /api/chat endpoint; most recent Ollama supports chat roles.

    Example models: 'llama3.1', 'qwen2.5:14b', 'mistral:instruct'
    """
    base_url: str = "http://localhost:11434"
    timeout_s: int = 600
    # If you prefer non-streaming responses; we stitch streamed chunks anyway.
    stream: bool = True

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
            "mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "num_gqa",
            "num_gpu", "num_thread", "repeat_last_n", "repeat_penalty",
            "seed", "stop", "tfs_z", "top_k", "top_p",
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
        with requests.post(url, json=payload, timeout=self.timeout_s, stream=self.stream) as r:
            r.raise_for_status()
            if not self.stream:
                data = r.json()
                return data.get("message", {}).get("content", "")

            parts: List[str] = []
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
                    break
            return "".join(parts)


# Proprietary 


@dataclass
class OpenAIClient(LLMClient):
    """
    Client for OpenAI API compatible endpoints (including Azure).

    - Reads `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` from environment.
    - `api_key` and `base_url` can be passed explicitly to override env vars.
    """
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = field(default=None)
    _client: Any = field(init=False, repr=False)

    def __post_init__(self):
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("`api_key` not provided and `OPENAI_API_KEY` env var not set.")
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
        return response.choices[0].message.content or ""




@dataclass
class GeminiClient(LLMClient):
    """
    Client for Google's Gemini models via the google-generativeai SDK.

    - Reads `GEMINI_API_KEY` from the environment.
    - `api_key` can be passed explicitly to override env var.
    """
    api_key: Optional[str] = field(default=None, repr=False)
    _model: Any = field(init=False, repr=False)

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("`api_key` not provided and `GEMINI_API_KEY` env var not set.")
        genai.configure(api_key=key)

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API."""
        self._model = genai.GenerativeModel(model)
        # The Gemini API expects a list of content blobs, not a role-based chat history.
        # We can simulate this by concatenating the messages.
        full_prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        sig = inspect.signature(self._model.generate_content)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        response = self._model.generate_content(
            full_prompt,
            generation_config=generation_config,
            **supported_kwargs,
        )
        return response.text