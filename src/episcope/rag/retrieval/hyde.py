from __future__ import annotations

import logging
from typing import List, Optional

from episcope.clients import LLMClient, OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_HYDE_PROMPT = (
    "You are a helpful research assistant. "
    "Write a short, hypothetical paragraph that could answer the following question. "
    "Do not include any preamble or meta-commentary. "
    "Focus on providing a direct, substantive response as if it were an excerpt from a relevant document.\n\n"
    "Question: {query}"
)


class HYDE:
    """
    Unified HYDE implementation used across the EpiScope project.

    A HYDE instance generates hypothetical documents from a user
    query. These documents can then be embedded alongside the query
    itself to improve retrieval performance.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = "llama3.2:1b",
        prompt_template: str = DEFAULT_HYDE_PROMPT,
    ) -> None:
        """
        Create a new HYDE generator.

        Args:
            client: An LLMClient instance. Defaults to OllamaClient.
            model: Name of the LLM model to use for generation.
            prompt_template: The template for the generation prompt.
        """
        self.client = client or OllamaClient()
        self.model = model
        self.prompt_template = prompt_template

    def generate(self, query: str) -> str:
        """
        Generate a single hypothetical paragraph for `query`.

        Args:
            query: The user’s information query.

        Returns:
            A single paragraph of generated text. Returns an empty
            string if generation fails.
        """
        prompt = self.prompt_template.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat(messages, model=self.model, temperature=0.5)
            logger.debug("Generated hypothetical document: %s...", response[:100])
            return response
        except Exception as e:
            logger.error("Error during HYDE generation: %s", e)
            return ""

    def generate_multiple(self, query: str, num_docs: int = 1) -> List[str]:
        """
        Generate multiple hypothetical documents for better retrieval.

        Args:
            query: The user’s information query.
            num_docs: Number of documents to generate.

        Returns:
            A list of generated documents.
        """
        # For now, this is a simple loop. In the future, we could
        # add logic to generate diverse responses.
        return [self.generate(query) for _ in range(num_docs)]
