from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from episcope.clients import GeminiClient, OllamaClient, OpenAIClient, OpenRouterClient
from episcope.rag.generation.llm_generator import LLMGenerator
from episcope.schemas import PaperMetadata
from episcope.workflows.classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
)
from episcope.workflows.classification.parsing import ClassificationResponseParser
from episcope.workflows.classification.prompting import ClassificationPromptBuilder
from episcope.workflows.classification.runner import ClassificationRunner


def build_llm_generator(
    *,
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> LLMGenerator:
    if provider == "gemini":
        client = GeminiClient()
    elif provider == "openrouter":
        client = OpenRouterClient()
    elif provider == "openai":
        client = OpenAIClient()
    elif provider == "ollama":
        client = OllamaClient()
    else:
        raise ValueError(
            f"Unknown provider={provider!r}. Use one of gemini, openrouter, openai, ollama."
        )
    return LLMGenerator(client=client, model=model, temperature=temperature)


class MetadataOnlyLLMBaseline:
    """Zero-shot LLM baseline using metadata and the label protocol, no retrieval."""

    name = "metadata_llm"

    def __init__(
        self,
        *,
        classifier_kind: str,
        generator: LLMGenerator,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.config = classifier_config(classifier_kind)
        self.prompt_builder = ClassificationPromptBuilder(self.config)
        self.response_parser = ClassificationResponseParser(self.config)
        self.runner = ClassificationRunner(
            generator=generator,
            config=self.config,
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
        )

    def predict(
        self,
        paper_id: str,
        metadata: PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        attempt = self.runner.run(metadata, chunks=[])
        return BaselinePrediction(
            result=attempt.result,
            raw_response=attempt.raw_response,
            prompt_messages=attempt.messages,
        )


def usage_snapshot_from_baseline(baseline: Any) -> Any | None:
    runner = getattr(baseline, "runner", None)
    generator = getattr(runner, "generator", None)
    client = getattr(generator, "client", None)
    usage_snapshot = getattr(client, "usage_snapshot", None)
    if callable(usage_snapshot):
        return usage_snapshot()
    return None


def usage_delta(after: Any, before: Any) -> dict[str, int]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "call_count",
    )
    return {
        field: int(getattr(after, field, 0) or 0) - int(getattr(before, field, 0) or 0)
        for field in fields
    }
