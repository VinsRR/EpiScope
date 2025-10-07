"""
core/hyde.py
================

This module centralises the implementation of the HYDE technique used
across both the Explorer (retrieval‑augmented generation) and
PrecisionMiner pipelines. HYDE stands for "hypothetical document
expansion" and improves retrieval performance by synthesising
contextual paragraphs from a user query. These paragraphs are
appended to the original query before embedding, leading to richer
document retrieval.

The HYDE class exposed here is a superset of the functionality found
in the original `retrieve/utils.py` and `parse/querying/hyde.py` classes.
By consolidating the implementation in a single place we avoid code
duplication and allow both pipelines to share improvements and bug
fixes.

Usage:

    from episcope.core.hyde import HYDE

    hyde = HYDE(model_name="llama3:8b")
    # Generate a single hypothetical paragraph
    paragraph = hyde.generate("What is R0 in epidemiology?")
    # Generate multiple paragraphs with optional paper context
    docs = hyde.generate_multiple("incubation period influenza", paper_title="Influenza Study", num_docs=3)
    # Enhance a query by appending generated paragraphs
    enhanced = hyde.enhance_query("serial interval COVID‑19")

The class also exposes additional helpers to generate style‑matched
documents and query the underlying model for metadata.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict, Any

import ollama

logger = logging.getLogger(__name__)


class HYDE:
    """
    Unified HYDE implementation used across the EpiScope project.

    A HYDE instance generates hypothetical documents from a user
    query. These documents can then be embedded alongside the query
    itself to improve retrieval performance. The implementation here
    merges the simple behaviour previously found in
    ``retrieve/utils.py`` with the more sophisticated, context‑aware
    prompts defined in ``parse/querying/hyde.py``.
    """

    def __init__(self, model_name: str = "tinyllama:1.1b") -> None:
        """Create a new HYDE generator.

        Args:
            model_name: Name of the Ollama model used for generation.
        """
        self.model_name = model_name

        # Domain‑specific prompt templates.  These may be extended
        # externally or overridden at runtime.  They are provided as
        # instance state rather than global constants to avoid shared
        # mutable state across threads or processes.
        self.domain_templates: Dict[str, str] = {
            "general": "Research findings indicate that {query}..."
        }

    # ------------------------------------------------------------------
    # Core generation methods
    #
    # The following public methods expose the core functionality of
    # HYDE.  They wrap the underlying Ollama API and provide
    # convenience helpers for query enhancement and style matching.

    def generate(self, query: str, paper_title: Optional[str] = None,
                 domain: Optional[str] = None) -> str:
        """Generate a single hypothetical paragraph for ``query``.

        Args:
            query: The user’s information query.
            paper_title: Optional paper title to bias generation towards
                content from a specific document.
            domain: Optional domain context (e.g. medical, physics).

        Returns:
            A single paragraph of generated text.  Returns a fallback
            message if generation fails.
        """
        docs = self.generate_multiple(query, paper_title, domain, num_docs=1)
        return docs[0] if docs else "Generation failed."

    def generate_multiple(self, query: str, paper_title: Optional[str] = None,
                          domain: Optional[str] = None, num_docs: int = 3) -> List[str]:
        """Generate multiple hypothetical documents for better retrieval.

        Args:
            query: The user’s information query.
            paper_title: Optional title of the paper for context.
            domain: Optional domain context for better prompts.
            num_docs: Number of documents to generate.

        Returns:
            A list of generated documents.  A fallback message is
            returned on failure.
        """
        results: List[str] = []
        for i in range(num_docs):
            prompt = self._construct_prompt(query, paper_title, domain, variant=i)
            try:
                # Adjust temperature slightly across runs to introduce
                # diversity.  This mirrors the behaviour of the more
                # sophisticated implementation.
                temperature = 0.4 + (i * 0.15)
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": min(temperature, 0.8),
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    },
                )
                content = response.get("message", {}).get("content", "").strip()
                generated_text = self._clean_preamble(content)
                if generated_text and generated_text != "Generation failed.":
                    results.append(generated_text)
                    logger.debug(
                        "Generated hypothetical document %d: %s...",
                        i + 1,
                        generated_text[:100],
                    )
            except Exception as e:
                logger.error("Error during HYDE generation %d: %s", i + 1, e)
                continue
        return results if results else ["Generation failed."]

    def enhance_query(self, query: str, paper_title: Optional[str] = None,
                      domain: Optional[str] = None, num_docs: int = 2) -> str:
        """Enhance a query by appending generated hypothetical documents.

        Args:
            query: Original query.
            paper_title: Optional paper title for context.
            domain: Optional domain context.
            num_docs: Number of hypothetical documents to generate.

        Returns:
            The enhanced query containing the original query and
            generated documents.  If generation fails, the original
            query is returned unchanged.
        """
        hypothetical_docs = self.generate_multiple(query, paper_title, domain, num_docs)
        valid_docs = [doc for doc in hypothetical_docs if doc != "Generation failed."]
        if not valid_docs:
            logger.warning("All HYDE generations failed; returning original query")
            return query
        parts = [query] + valid_docs
        enhanced = " ".join(parts)
        logger.debug(
            "Enhanced query from %d to %d characters", len(query), len(enhanced)
        )
        return enhanced

    def generate_with_style_examples(self, query: str, style_examples: List[str],
                                     domain: Optional[str] = None) -> str:
        """Generate a document matching the style of given examples.

        Args:
            query: Information query to address.
            style_examples: Example excerpts to imitate.
            domain: Optional domain context.

        Returns:
            A generated paragraph matching the style.  Returns a fallback
            message on failure.
        """
        # Limit examples to avoid token limits
        examples_text = "\n\n".join(
            [
                f"Example {i+1}: {example[:200]}..." if len(example) > 200 else f"Example {i+1}: {example}"
                for i, example in enumerate(style_examples[:3])
            ]
        )
        prompt = (
            "Here are examples of the writing style from target documents:\n\n"
            f"{examples_text}\n\n"
            f"Now write a paragraph in the same style that would be relevant to: '{query}'. "
            "Match the technical level, terminology, and format of the examples. "
            "Write as if excerpting from an actual document"
            f" in {domain}" if domain else "" + ". "
            "Start directly with substantive content."
        )
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            )
            content = response.get("message", {}).get("content", "").strip()
            generated_text = self._clean_preamble(content)
            logger.debug(
                "Generated style‑matched document: %s...", generated_text[:100]
            )
            return generated_text
        except Exception as e:
            logger.error("Error during style‑based HYDE generation: %s", e)
            return "Generation failed."

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the configured model.

        Returns:
            A dictionary with model metadata such as parameter size, family,
            and quantisation level.  On failure a dictionary with an
            ``error`` field is returned.
        """
        try:
            info = ollama.show(self.model_name)
            return {
                "model_name": self.model_name,
                "parameters": info.get("details", {}).get("parameter_size"),
                "family": info.get("details", {}).get("family"),
                "quantization": info.get("details", {}).get("quantization_level"),
            }
        except Exception as e:
            logger.error("Could not get model info: %s", e)
            return {"model_name": self.model_name, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal helper methods

    def _construct_prompt(self, query: str, paper_title: Optional[str],
                          domain: Optional[str], variant: int) -> str:
        """Construct a domain‑aware prompt for document generation.

        Strategies vary across invocations to produce diverse output.

        Args:
            query: The user’s query.
            paper_title: Optional title of the paper for context.
            domain: Optional domain string.
            variant: A zero‑based index used to select the prompt strategy.

        Returns:
            A prompt string to pass to the LLM.
        """
        strategies = ["excerpt", "methodology", "results"]
        strategy = strategies[variant % len(strategies)]
        # If a paper title is provided we bias the prompt accordingly
        if paper_title:
            if strategy == "excerpt":
                return (
                    f"Extract from research paper: '{paper_title}'\n\n"
                    f"[The following section discusses {query}]\n\n"
                    "Write the next paragraph as it would appear in this paper. "
                    "Use technical language and specific details."
                )
            elif strategy == "methodology":
                return (
                    f"From the Methods section of '{paper_title}':\n\n"
                    f"To investigate {query}, we employed... "
                    "Continue writing as if describing the methodology."
                )
            else:  # results
                return (
                    f"From the Results section of '{paper_title}':\n\n"
                    f"Our analysis of {query} revealed... "
                    "Continue with specific findings and data."
                )
        # Otherwise, fall back to domain templates
        base_prompt = (
            self.domain_templates.get(domain, self.domain_templates["general"]).format(query=query)
        )
        if strategy == "methodology":
            base_prompt += " Our methodology involved..."
        elif strategy == "results":
            base_prompt += " The results showed that..."
        return (
            f"{base_prompt} "
            "Write as if excerpting from an actual research document. "
            "Use technical terminology and specific details. "
            "Do not include meta‑commentary about the document itself."
        )

    def _clean_preamble(self, text: str) -> str:
        """Remove common preambles and meta commentary from generated text."""
        patterns = [
            r'^This (?:paper|study|research|document|investigation) .*?\. ',
            r'^The (?:paper|study|research|document|investigation) .*?\. ',
            r'^In this (?:paper|study|research|document) .*?\. ',
            r'^According to (?:this|the) .*?\. ',
            r'^Based on (?:this|the) .*?\. ',
            r'^Our (?:paper|study|research|investigation) .*?\. ',
            r'^Here (?:we|is) .*?\. ',
            r'^\[.*?\]\s*',  # Remove section markers like [Methods]
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        return cleaned.strip()