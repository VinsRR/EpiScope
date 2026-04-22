from __future__ import annotations

from episcope.workflows.precision_miner.parsing import PrecisionMinerResponseParser


def test_precision_miner_parser_accepts_fenced_json() -> None:
    response = """```json
    {
      "description": "One source found",
      "items": [
        {
          "name": "NHANES",
          "url": null,
          "explanation": "The study used NHANES data.",
          "raw_text": "We used NHANES."
        }
      ]
    }
    ```"""

    result = PrecisionMinerResponseParser.parse(response)

    assert result.description == "One source found"
    assert len(result.items) == 1
    assert result.items[0].name == "NHANES"


def test_precision_miner_fallback_contains_error_message() -> None:
    error = ValueError("bad json")

    result = PrecisionMinerResponseParser.fallback(error)

    assert result.items == []
    assert "bad json" in result.description
