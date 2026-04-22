from __future__ import annotations

from episcope.workflows.precision_miner.schemas import ExtractionResult


class PrecisionMinerResponseParser:
    """Parse generator output into the extraction result schema."""

    @staticmethod
    def parse(response: str) -> ExtractionResult:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return ExtractionResult.model_validate_json(cleaned)

    @staticmethod
    def fallback(error: Exception) -> ExtractionResult:
        return ExtractionResult(description=f"Parsing/validation failed: {error}", items=[])
