from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import ValidationError

from episcope.workflows.classification.config import BaseClassifierConfig
from episcope.workflows.classification.schemas import ClassificationResult


class ClassificationParseError(Exception):
    """Raised when an LLM response cannot be parsed into a ClassificationResult."""


class ClassificationResponseParser:
    """Validate and normalize raw LLM JSON into ClassificationResult."""

    def __init__(self, config: BaseClassifierConfig) -> None:
        self.config = config

    def parse(self, response_content: str) -> ClassificationResult:
        raw = response_content.strip()
        parsed = self._parse_json(raw)
        classification = self._extract_classification(parsed)
        if not self.config.multi_label and len(classification) > 1:
            classification = classification[:1]
        extras = self._extract_extras(parsed)
        return ClassificationResult(
            classification=classification,
            confidence=float(parsed.confidence)
            if getattr(parsed, "confidence", None) is not None
            else 0.0,
            class_probabilities=parsed.class_probabilities or {},
            evidence={"reasoning": parsed.reasoning},
            extras=extras,
        )

    def create_fallback_result(self) -> ClassificationResult:
        n = len(self.config.category_labels)
        uniform = 1 / n if n else 0.0
        return ClassificationResult(
            classification=self.config.default_classification,
            confidence=0.0,
            class_probabilities={
                label: uniform for label in self.config.category_labels.values()
            },
            evidence={"reasoning": "Classification failed; defaulting to unclear."},
            extras={},
        )

    def _parse_json(self, raw: str) -> Any:
        try:
            return self.config.output_schema.model_validate_json(raw)
        except Exception:
            pass

        candidate = self._extract_json_candidate(raw)
        if candidate is None:
            raise ClassificationParseError("No JSON object found in model output.")

        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ClassificationParseError(
                f"Extracted JSON is not valid JSON: {exc}"
            ) from exc

        try:
            return self.config.output_schema.model_validate_json(candidate)
        except ValidationError as exc:
            raise ClassificationParseError(
                f"JSON does not match schema: {exc}"
            ) from exc
        except Exception as exc:
            raise ClassificationParseError(
                f"Unexpected validation error: {exc}"
            ) from exc

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[str]:
        for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth, in_string, escape = 0, False, False
        for index, char in enumerate(text[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
        return None

    def _extract_classification(self, parsed: Any) -> list[Any]:
        if hasattr(parsed, "classification"):
            raw_cls = parsed.classification
            if isinstance(raw_cls, list):
                mapped = [
                    self.config.classification_mapping[item]
                    for item in raw_cls
                    if item in self.config.classification_mapping
                ]
                return mapped or self.config.default_classification

            mapped = self.config.classification_mapping.get(raw_cls)
            return (
                [mapped] if mapped is not None else self.config.default_classification
            )

        if hasattr(parsed, "primary_label"):
            return [parsed.primary_label]

        raise ClassificationParseError(
            "Parsed output has neither 'classification' nor 'primary_label'."
        )

    @staticmethod
    def _extract_extras(parsed: Any) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        if hasattr(parsed, "extras") and parsed.extras:
            extras = (
                parsed.extras.model_dump()
                if hasattr(parsed.extras, "model_dump")
                else dict(parsed.extras)
            )
        if hasattr(parsed, "secondary_labels") and parsed.secondary_labels:
            extras["secondary_labels"] = parsed.secondary_labels
        return extras
