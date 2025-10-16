"""
Unit tests for the LLMExtractor class.

These tests validate helper functions and core extraction logic in
``episcope.generate.llm_extractor``.  External model calls
are replaced with dummy functions to avoid network dependencies.
"""
import unittest
from typing import Any, Dict, List, Sequence, Mapping, Optional

from episcope.rag.generation.llm_extractor import LLMExtractor, _extract_json_blob
from episcope.rag.generation.clients import LLMClient


class MockLLMClient(LLMClient):
    def __init__(self, response_text: str):
        self.response_text = response_text

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        return self.response_text


class TestLLMExtractor(unittest.TestCase):
    """Tests for JSON extraction and data source extraction."""

    def test_extract_json_blob(self) -> None:
        """_extract_json_blob should return the first JSON object in a string."""
        text = "prefix {\"key\": 1, \"value\": \"a\"} suffix"
        self.assertEqual(_extract_json_blob(text), '{"key": 1, "value": "a"}')

    def test_extract_data_sources_parses_valid_json(self) -> None:
        """extract_data_sources should parse JSON and construct an ExtractionResult."""
        # Mock client that returns a valid JSON string
        mock_response = (
            '{"data_sources_description": "A data source description",'
            ' "data_sources": [{"source_name": "Dataset1", "url": "N/A", "explanation": "Used for analysis", "section_found": "Methods"}],'
            ' "references": []}'
        )
        mock_client = MockLLMClient(response_text=mock_response)

        extractor = LLMExtractor(client=mock_client)
        # Create minimal metadata object with required attributes
        class Meta:
            title = "Test Title"
            abstract = "Test abstract"
        metadata = Meta()

        result, conf = extractor.generate(
            relevant_chunks=[{"text": "Example chunk."}],
            references=[],
            paper_type="data_analysis",
            metadata=metadata,
            query="What data sources were used?"
        )
        self.assertEqual(result.data_sources_description, "A data source description")
        self.assertEqual(len(result.data_sources), 1)
        self.assertEqual(result.data_sources[0].source_name, "Dataset1")


if __name__ == "__main__":
    unittest.main()