from __future__ import annotations

import json
from typing import Dict, List

from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.precision_miner.config import PrecisionMinerConfig
from episcope.workflows.precision_miner.schemas import ExtractionResultSchema


class PrecisionMinerPromptBuilder:
    """Build prompts for extraction-oriented precision miner workflows."""

    def __init__(self, config: PrecisionMinerConfig) -> None:
        self.config = config

    def build_messages(
        self,
        metadata: PaperMetadata,
        chunks: List[SearchResult],
    ) -> List[Dict[str, str]]:
        schema = ExtractionResultSchema.model_json_schema()
        user_prompt = self.config.user_prompt_template.format(
            title=metadata.title,
            abstract=metadata.abstract or "N/A",
            keywords=", ".join(metadata.keywords or []),
            chunks_info=self.format_chunks(chunks),
            schema=json.dumps(schema),
        )
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def format_chunks(chunks: List[SearchResult]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)
