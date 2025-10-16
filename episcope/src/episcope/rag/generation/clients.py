"""
LLM clients for interfacing with different providers (Ollama, OpenAI, etc.).

This module provides a provider-agnostic interface `LLMClient` and concrete
implementations for various services. This allows generators and other
components to be written without being tied to a specific LLM provider.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Protocol, Optional, Mapping
from dataclasses import dataclass
import json

import requests  # only used by the Ollama client; swap as needed


# ---------- Provider-agnostic interface ----------

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


# ---------- Ollama implementation ----------

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
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "options": {"temperature": temperature},
            "stream": self.stream,
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

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


# ---------- Optional: placeholder for other providers ----------

@dataclass
class OpenAIClient(LLMClient):
    """
    Example stub to show how you'd extend to other providers.
    Replace with real implementation using openai>=1.0 SDK, etc.
    """
    api_key: str
    base_url: Optional[str] = None  # e.g., Azure/OpenAI-compatible endpoints

    def chat(  # type: ignore[override]
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Provide an OpenAI/Azure implementation if needed.")

@dataclass
class GeminiClient(LLMClient):
    """
    Example stub to show how you'd extend to other providers.
    Replace with real implementation using google.generativeai>=0.3.0 SDK, etc.
    """
    api_key: str
    base_url: Optional[str] = None  # e.g., Gemini-compatible endpoints

    def chat(  # type: ignore[override]
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Provide a Gemini implementation if needed.")
