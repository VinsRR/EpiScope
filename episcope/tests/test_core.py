"""
Unit tests for core modules of the EpiScope package.

These tests exercise simple behaviours of the provenance and provider
abstractions without requiring any external services.  They use the
built‑in ``unittest`` framework and standard library mocks to
substitute external dependencies.
"""
import sys
import types
import unittest
from unittest import mock


class TestProvenance(unittest.TestCase):
    """Ensure Evidence and Provenance dataclasses behave as expected."""

    def test_evidence_fields(self) -> None:
        from episcope.core.provenance import Evidence

        ev = Evidence(
            paper_id="paper123",
            snippet="some text",
            section="Introduction",
            index_version="v1",
            model_id="modelX",
            prompt_id="prompt1",
        )
        self.assertEqual(ev.paper_id, "paper123")
        self.assertEqual(ev.snippet, "some text")
        self.assertEqual(ev.section, "Introduction")
        self.assertEqual(ev.index_version, "v1")
        self.assertEqual(ev.model_id, "modelX")
        self.assertEqual(ev.prompt_id, "prompt1")

    def test_provenance_contains_evidence(self) -> None:
        from episcope.core.provenance import Evidence, Provenance

        ev1 = Evidence(paper_id="p1", snippet="s1")
        ev2 = Evidence(paper_id="p2", snippet="s2")
        prov = Provenance(answer="answer text", evidences=[ev1, ev2])
        self.assertEqual(prov.answer, "answer text")
        self.assertEqual(len(prov.evidences), 2)
        self.assertIn(ev1, prov.evidences)
        self.assertIn(ev2, prov.evidences)


class TestProviderFactory(unittest.TestCase):
    """Test provider factory registration and instantiation."""

    def test_local_provider_registration(self) -> None:
        """Ensure the local provider can be instantiated without Ollama present.

        We monkey‑patch the ``ollama`` module before importing the
        provider to avoid import errors.  Only the factory lookup and
        type of the instance are validated; the chat method is not
        invoked here.
        """
        import types
        # Inject a dummy ollama module into sys.modules
        dummy = types.SimpleNamespace(chat=lambda *a, **k: None)
        with mock.patch.dict(sys.modules, {"ollama": dummy}):
            from episcope.core.provider import ProviderFactory
            from episcope.providers.local import LocalProvider
            provider = ProviderFactory.create("local", model="dummy")
            self.assertIsInstance(provider, LocalProvider)

    def test_unknown_provider_raises(self) -> None:
        from episcope.core.provider import ProviderFactory

        with self.assertRaises(ValueError):
            ProviderFactory.create("nonexistent")


if __name__ == "__main__":
    unittest.main()