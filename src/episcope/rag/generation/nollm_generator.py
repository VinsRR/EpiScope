"""Simple generator for synthesising answers from retrieved contexts.

This generator concatenates the retrieved contexts and treats the
resulting string as the answer.  It constructs provenance metadata
for each context to maintain transparency and traceability.  This
implementation avoids calling any external LLM and is intended for
testing and demonstration purposes when heavyweight models are not
available.  It implements the :class:`AbstractGenerator` interface.
"""

from __future__ import annotations

from typing import Any, Sequence, Dict, List

from episcope.rag.interfaces import AbstractGenerator
from episcope.rag.provenance import Provenance, Evidence


class NoLLMGenerator(AbstractGenerator):
    """Concatenate context snippets into an answer string.

    This generator simply joins the ``content`` fields of the
    provided contexts separated by two newlines.  Each context
    contributes an evidence entry.  No language model inference is
    performed.
    """

    def generate(self, contexts: Sequence[Dict[str, Any]], **kwargs: Any) -> Provenance:
        answer_parts: List[str] = []
        evidences: List[Evidence] = []
        for ctx in contexts:
            content = ctx.text or ""
            answer_parts.append(content)
            paper_id = ctx.paper_id
            section = ctx.section_type
            evidences.append(
                Evidence(
                    paper_id=paper_id,
                    snippet=content,
                    section=section,
                    index_version=None,
                    model_id="simple-generator",
                    prompt_id="concatenate",
                )
            )
        full_answer = "\n\n".join(answer_parts).strip()
        return Provenance(answer=full_answer, evidences=evidences)
