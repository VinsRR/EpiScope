from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.utils import PromptMessage, result_score


# Matches a "**Relevant Extracts...**" heading followed by any subsequent blank
# or whitespace-only lines, stopping just before the next non-blank line.
# Used to remove the (now-empty) extracts block from the rendered user prompt
# when the caller passes no chunks (e.g. the metadata-only LLM baseline). The
# main RAG pipeline always passes retrieved chunks, so the stripping path is
# never exercised there.
_EXTRACTS_BLOCK_RE = re.compile(
    r"\*\*Relevant Extracts[^\n]*\*\*[ \t]*\n(?:[ \t]*\n)*",
    re.IGNORECASE,
)


class ClassificationPromptBuilder:
    """Build prompts for classification from metadata and selected evidence."""

    def __init__(self, config: BaseClassifierConfig) -> None:
        self.config = config

    def build_initial_prompt(
        self,
        metadata: PaperMetadata,
        chunks: List[SearchResult],
    ) -> List[PromptMessage]:
        schema = self.config.output_schema.model_json_schema()
        categories = "\n".join(
            f"{key} – {value}" for key, value in self.config.category_labels.items()
        )
        user_prompt = self.config.user_prompt_template.format(
            categories=categories,
            title=metadata.title,
            abstract=metadata.abstract or "N/A",
            keywords=", ".join(metadata.keywords or []),
            chunks_info=self.format_chunks_for_prompt(chunks),
            schema=json.dumps(schema),
            **self.config.extra_output_fields,
        )
        if not chunks:
            # Without retrieved chunks the templates' "**Relevant Extracts:**"
            # header is followed by an empty payload, which has been observed
            # to confuse generators into protesting about the empty section
            # rather than answering. Drop the header so the prompt looks like
            # a clean metadata-only request.
            user_prompt = _EXTRACTS_BLOCK_RE.sub("", user_prompt)
        system_prompt = self.config.system_prompt.format(
            n_categories=len(self.config.category_labels),
            category_labels=", ".join(self.config.category_labels.values()),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def format_chunks_for_prompt(self, chunks: List[SearchResult]) -> str:
        """Group chunks by category for prompt readability."""
        by_category: Dict[str, List[SearchResult]] = defaultdict(list)
        for chunk in chunks:
            category = chunk.artifacts.get("category", "unknown")
            by_category[category].append(chunk)

        parts: List[str] = []
        for category, cat_chunks in by_category.items():
            avg = np.mean([result_score(chunk) for chunk in cat_chunks])
            lines = [f"\n{category.upper()} (avg {avg:.3f}):"]
            lines += [
                f"  • {result_score(chunk):.3f}: {chunk.text}" for chunk in cat_chunks
            ]
            parts.append("\n".join(lines))
        return "\n".join(parts)
