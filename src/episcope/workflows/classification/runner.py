from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from episcope.rag.generation.base import Generator
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.output import CompletionSample
from episcope.workflows.classification.parsing import (
    ClassificationParseError,
    ClassificationResponseParser,
)
from episcope.workflows.classification.prompting import ClassificationPromptBuilder
from episcope.workflows.classification.schemas import ClassificationResult
from episcope.workflows.classification.utils import PromptMessage

logger = logging.getLogger(__name__)


@dataclass
class LLMAttemptResult:
    """Outcome of a single generation + parse attempt."""

    result: ClassificationResult
    raw_response: str
    messages: List[PromptMessage]
    all_samples: List[CompletionSample] = field(default_factory=list)


class ClassificationRunner:
    """Run the LLM classification loop with validation retries."""

    def __init__(
        self,
        generator: Generator,
        config: BaseClassifierConfig,
        prompt_builder: ClassificationPromptBuilder,
        response_parser: ClassificationResponseParser,
    ) -> None:
        self.generator = generator
        self.config = config
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser

    def run(
        self,
        metadata: PaperMetadata,
        chunks: List[SearchResult],
        *,
        expect_no_chunks: bool = False,
    ) -> LLMAttemptResult:
        if not chunks and not expect_no_chunks:
            logger.warning("No relevant chunks found for classification.")

        messages = self.prompt_builder.build_initial_prompt(metadata, chunks)
        samples: List[CompletionSample] = []
        last_raw: Optional[str] = None

        for attempt in range(1, self.config.max_validation_retries + 1):
            try:
                generation = self.generator.generate(
                    contexts=[[chunk.text for chunk in chunks]],
                    message_builder=lambda **_: messages,
                    format="json",
                )
                last_raw = generation.answer
            except Exception as exc:
                logger.warning("LLM generation attempt %d failed: %s", attempt, exc)
                continue

            try:
                result = self.response_parser.parse(last_raw)
                samples.append(
                    CompletionSample(
                        messages=list(messages), completion=last_raw, parsed_ok=True
                    )
                )
                return LLMAttemptResult(
                    result=result,
                    raw_response=last_raw,
                    messages=list(messages),
                    all_samples=samples,
                )
            except ClassificationParseError as exc:
                logger.warning(
                    "Parse attempt %d/%d failed: %s",
                    attempt,
                    self.config.max_validation_retries,
                    exc,
                )
                logger.debug("Raw response:\n%s", last_raw)
                samples.append(
                    CompletionSample(
                        messages=list(messages), completion=last_raw, parsed_ok=False
                    )
                )
                if attempt < self.config.max_validation_retries:
                    messages = self._append_correction_turn(messages, last_raw, exc)

        logger.error(
            "Classification failed after %d attempt(s). Returning fallback result.",
            self.config.max_validation_retries,
        )
        return LLMAttemptResult(
            result=self.response_parser.create_fallback_result(),
            raw_response=last_raw or "",
            messages=list(messages),
            all_samples=samples,
        )

    @staticmethod
    def _append_correction_turn(
        messages: List[PromptMessage],
        bad_response: str,
        error: Exception,
    ) -> List[PromptMessage]:
        return messages + [
            {"role": "assistant", "content": bad_response},
            {
                "role": "user",
                "content": (
                    f"Your previous response was not valid. Error: {error}\n"
                    "Please return ONLY a single valid JSON object that matches the schema. "
                    "Do not include any extra text, markdown fences, or explanation."
                ),
            },
        ]
