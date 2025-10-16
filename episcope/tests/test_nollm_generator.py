"""Tests for the SimpleGenerator.

These tests confirm that the ``SimpleGenerator`` concatenates
contexts into an answer and constructs provenance entries for each
context.  The generator does not invoke any external LLMs.
"""

import unittest
from episcope.rag.generation.nollm_generator import NoLLMGenerator


class TestNoLLMGenerator(unittest.TestCase):
    def test_generate_concatenates_contexts(self) -> None:
        generator = NoLLMGenerator()
        contexts = [
            type('obj', (object,),{"text": "First snippet.", "paper_id": "A", "section_type": "Intro"}),
            type('obj', (object,),{"text": "Second snippet.", "paper_id": "B", "section_type": "Methods"})
        ]
        prov = generator.generate(contexts)
        # The answer should be concatenated with a blank line
        self.assertEqual(prov.answer, "First snippet.\n\nSecond snippet.")
        # There should be two evidences
        self.assertEqual(len(prov.evidences), 2)
        # Evidence paper_ids should match
        self.assertEqual(prov.evidences[0].paper_id, "A")
        self.assertEqual(prov.evidences[1].paper_id, "B")


if __name__ == "__main__":
    unittest.main()