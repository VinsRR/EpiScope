from __future__ import annotations

import logging
from typing import Optional, Union

from episcope.clients import LLMClient, OllamaClient
from episcope.rag.retrieval.components import QueryTransformer
from episcope.rag.retrieval.hyde import HYDE

logger = logging.getLogger(__name__)


class HyDEQueryTransformer(QueryTransformer):
    """Augments a query with one or more hypothetical supporting passages."""

    def __init__(
        self,
        hyde: Optional[Union[bool, HYDE]] = None,
        llm_client: Optional[LLMClient] = None,
        num_docs: int = 1,
    ) -> None:
        self.num_docs = num_docs

        if hyde is True:
            self.hyde: Optional[HYDE] = HYDE(
                client=llm_client if llm_client else OllamaClient()
            )
        elif isinstance(hyde, HYDE):
            self.hyde = hyde
        else:
            self.hyde = None

    def transform(self, query: str) -> str:
        if self.hyde is None:
            return query

        hypothetical_docs = self.hyde.generate_multiple(query, num_docs=self.num_docs)
        valid_docs = [
            doc for doc in hypothetical_docs if doc and "failed" not in doc.lower()
        ]
        if not valid_docs:
            return query

        transformed = f"{query}\n\n" + "\n\n".join(valid_docs)
        logger.debug("HYDE enhanced query: %s", transformed)
        return transformed
