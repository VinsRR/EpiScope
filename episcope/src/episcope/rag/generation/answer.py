"""
generate/answer.py

High level answer generation orchestrator.

This module coordinates retrieval, generation and provenance
construction.  It selects a retrieval method via ``RAGFactory`` and a
language model via ``ProviderFactory``.  The generated answer and its
supporting evidences are returned in a ``Provenance`` object.
"""
from __future__ import annotations

from typing import List, Optional

from episcope.rag.retrieval import RAGFactory
from episcope.rag.provenance import Evidence, Provenance
from episcope.rag.providers.base import ProviderFactory
from episcope.settings import CONFIG


class AnswerGenerator:
    """Orchestrates retrieval, LLM generation and provenance construction."""

    def __init__(self, rag_method: str = "text") -> None:
        self.rag_method_name = rag_method
        self.rag = RAGFactory.get(rag_method)
        self.provider = ProviderFactory.create(CONFIG.provider.provider, model=CONFIG.provider.model, api_key=CONFIG.provider.api_key)

    def answer_question(self, query: str, top_k: int = 5) -> Provenance:
        # Retrieve contexts
        contexts = self.rag.retrieve(query, top_k=top_k)
        # Build chat messages
        snippet_list = "\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(contexts))
        # Build system/user prompts from the configured RAG
        system_prompt = getattr(self.rag, "system_prompt", "You are a helpful assistant.")
        user_prompt_template = getattr(self.rag, "user_prompt", "Answer the question: {question}\n\n{snippet_list}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(question=query, num_snippets=len(contexts), snippet_list=snippet_list)},
        ]
        # Generate answer via provider
        answer = self.provider.chat(messages)
        # Construct evidences
        evidences: List[Evidence] = []
        for c in contexts:
            paper_id = str(c.get("id", c.get("file", "unknown")))
            snippet = c.get("content", c.get("text", ""))
            section = c.get("section_type", c.get("type", None))
            evidences.append(
                Evidence(
                    paper_id=paper_id,
                    snippet=snippet,
                    section=section,
                    index_version=None,
                    model_id=CONFIG.provider.model,
                    prompt_id=self.rag_method_name,
                )
            )
        return Provenance(answer=answer, evidences=evidences)

