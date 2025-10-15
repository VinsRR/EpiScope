"""
LLM-backed generator for synthesising answers from retrieved contexts.

This implementation builds a prompt from retrieved contexts and asks an LLM to
compose an answer. It is provider-agnostic via a simple LLMClient interface,
with a concrete Ollama client included by default (and easy to extend to other
providers, e.g., OpenAI, Azure, etc.).

It returns a Provenance object with the model output as `answer` and one
Evidence per input context to preserve traceability. The Evidence entries
record the model/prompt used but keep the snippet as the raw context text
(rather than LLM output), so downstream code can map answer → sources.

Implements the :class:`AbstractGenerator` interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Protocol, Optional, Mapping
from dataclasses import dataclass, field
import json
import time

import requests  # only used by the Ollama client; swap as needed

from episcope.rag.interfaces import AbstractGenerator
from episcope.rag.provenance import Provenance, Evidence


# ---------- Provider-agnostic interface ----------

class LLMClient(Protocol):
    """Minimal interface to support multiple LLM providers."""
    def generate(
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
class OllamaClient:
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

    def generate(
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
class OpenAIClient:
    """
    Example stub to show how you'd extend to other providers.
    Replace with real implementation using openai>=1.0 SDK, etc.
    """
    api_key: str
    base_url: Optional[str] = None  # e.g., Azure/OpenAI-compatible endpoints

    def generate(  # type: ignore[override]
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
class GeminiClient:
    """
    Example stub to show how you'd extend to other providers.
    Replace with real implementation using google.generativeai>=0.3.0 SDK, etc.
    """
    api_key: str
    base_url: Optional[str] = None  # e.g., Gemini-compatible endpoints

    def generate(  # type: ignore[override]
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Provide a Gemini implementation if needed.")

# ---------- Utility: safe context access ----------

def _get(ctx: Any, key: str, default: Any = None) -> Any:
    """Support both dict-like and attr-like contexts."""
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


# ---------- Prompt construction ----------

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful scientific assistant. Using ONLY the provided contexts, "
    "compose a fluent, concise answer. Do not invent facts. If the contexts "
    "lack information, say so explicitly. Do not include citations in the text; "
    "the caller will attach provenance separately."
)

def build_messages_from_contexts(
    contexts: Sequence[Dict[str, Any] | Any],
    user_question: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    joiner: str = "\n\n---\n\n",
) -> List[Dict[str, str]]:
    # Flatten contexts into a readable block
    blocks: List[str] = []
    for idx, ctx in enumerate(contexts):
        text = _get(ctx, "text", "") or _get(ctx, "content", "") or ""
        paper_id = _get(ctx, "paper_id", "")
        section = _get(ctx, "section_type", "") or _get(ctx, "section", "")
        header = f"[Context {idx+1}{f' | paper_id={paper_id}' if paper_id else ''}{f' | section={section}' if section else ''}]"
        blocks.append(f"{header}\n{text}")

    corpus = joiner.join(blocks).strip()
    # Compose user instruction
    question = (user_question or "Synthesize the best possible answer from the contexts above.").strip()
    user_prompt = (
        f"{question}\n\n"
        "Contexts:\n"
        f"{corpus}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------- The LLM-backed generator ----------

class LLMGenerator(AbstractGenerator):
    """
    Use an LLM to generate an answer from retrieved contexts.

    Parameters
    ----------
    client : LLMClient
        Provider client (defaults to Ollama).
    model : str
        Model identifier (e.g., 'llama3.1' for Ollama).
    temperature : float
        Decoding temperature (default 0.0 for determinism).
    max_tokens : Optional[int]
        Max tokens (provider-specific; Ollama uses `num_predict`).
    system_prompt : str
        System message steering the model.
    joiner : str
        Separator between context blocks in the prompt.
    model_id_tag : Optional[str]
        Tag stored in Evidence.model_id for traceability; defaults to provider+model.
    prompt_id_tag : str
        Tag stored in Evidence.prompt_id to record the prompt recipe.

    Notes
    -----
    - Returns `Provenance` with `answer` = model output and one Evidence per context.
    - Evidence.snippet contains the original context text for transparency.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        *,
        model: str = "llama3.2:1b",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        joiner: str = "\n\n---\n\n",
        model_id_tag: Optional[str] = None,
        prompt_id_tag: str = "llm-synthesis-v1",
    ) -> None:
        self.client = client or OllamaClient()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.joiner = joiner
        self.model_id_tag = model_id_tag or self._default_model_id_tag()
        self.prompt_id_tag = prompt_id_tag

    def _default_model_id_tag(self) -> str:
        # Try to infer provider name from class
        provider = type(self.client).__name__.replace("Client", "").lower()
        return f"{provider}:{self.model}"

    def generate(
        self,
        contexts: Sequence[Dict[str, Any]] | Sequence[Any],
        *,
        question: Optional[str] = None,
        extra_messages: Optional[Sequence[Mapping[str, str]]] = None,
        **kwargs: Any,
    ) -> Provenance:
        # 1) Build messages
        messages = build_messages_from_contexts(
            contexts,
            user_question=question,
            system_prompt=self.system_prompt,
            joiner=self.joiner,
        )
        if extra_messages:
            messages.extend(extra_messages)

        # 2) Call provider
        started = time.time()
        answer = self.client.generate(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_s = time.time() - started  # kept local; attach if you log metrics

        # 3) Build evidences (one per context)
        evidences: List[Evidence] = []
        for ctx in contexts:
            content = _get(ctx, "text", "") or _get(ctx, "content", "") or ""
            evidences.append(
                Evidence(
                    paper_id=_get(ctx, "paper_id", None),
                    snippet=content,
                    section=_get(ctx, "section_type", None) or _get(ctx, "section", None),
                    index_version=None,
                    model_id=self.model_id_tag,
                    prompt_id=self.prompt_id_tag,
                )
            )

        return Provenance(answer=answer.strip(), evidences=evidences)


