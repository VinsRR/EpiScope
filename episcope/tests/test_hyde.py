"""
Unit tests for the HYDE class.

This test module validates the behaviour of the unified HYDE
implementation found in ``episcope.core.hyde``.  The tests patch
external dependencies (Ollama) to avoid network calls and ensure
deterministic outputs.  They exercise the main generation methods
and query enhancement.

These tests use the built‑in ``unittest`` framework rather than
pytest to avoid requiring external test runners.
"""
import unittest
from unittest.mock import patch

# Stub external modules before importing HYDE.  Some parts of the
# codebase import ``ollama`` at module scope, which is unavailable in
# this test environment.  Provide a minimal stub to satisfy imports.
import sys
import types
sys.modules.setdefault(
    "ollama",
    types.SimpleNamespace(
        chat=lambda *args, **kwargs: {"message": {"content": ""}},
        show=lambda *args, **kwargs: {},
        Client=lambda *args, **kwargs: None,
    ),
)

from episcope.core.hyde import HYDE


class TestHYDE(unittest.TestCase):
    """Tests for the HYDE class."""

    def test_generate_returns_generated_text(self) -> None:
        """Ensure that generate() returns the cleaned content from the model."""
        # Patch ollama.chat to return a fixed response
        with patch("episcope.core.hyde.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "This is a synthetic paragraph."}}
            hyde = HYDE(model_name="tinyllama:1.1b")
            result = hyde.generate("What is R0 in epidemiology?")
            self.assertEqual(result, "This is a synthetic paragraph.")

    def test_enhance_query_appends_hypothetical(self) -> None:
        """Verify that enhance_query() appends generated documents to the query."""
        # Provide multiple paragraphs for generate_multiple
        def fake_chat(model: str, messages: list, options: dict) -> dict:
            # The prompt is not inspected here; return distinct content per call
            content = "Generated doc." if fake_chat.counter == 0 else "Another doc."
            fake_chat.counter += 1
            return {"message": {"content": content}}

        fake_chat.counter = 0
        with patch("episcope.core.hyde.ollama.chat", side_effect=fake_chat):
            hyde = HYDE(model_name="tinyllama")
            enhanced = hyde.enhance_query("Query", paper_title=None, domain=None, num_docs=2)
            # The enhanced string should contain the original query and both generated docs
            self.assertIn("Query", enhanced)
            self.assertIn("Generated doc.", enhanced)
            self.assertIn("Another doc.", enhanced)


if __name__ == "__main__":
    unittest.main()