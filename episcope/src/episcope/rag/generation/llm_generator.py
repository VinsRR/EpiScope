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

from typing import Any, Dict, List, Sequence, Protocol, Optional, Mapping, Callable
from dataclasses import dataclass, field
import json
import time

from episcope.rag.interfaces import AbstractGenerator
from episcope.rag.provenance import Provenance, Evidence
from episcope.clients import LLMClient, OllamaClient

# Utility: safe context access 

def _get(ctx: Any, key: str, default: Any = None) -> Any:
    """Support both dict-like and attr-like contexts."""
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)



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
        message_builder: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Provenance:
        # 1) Build messages
        if message_builder:
            messages = message_builder(contexts=contexts, question=question, **kwargs)
        else:
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
        answer = self.client.chat(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
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
