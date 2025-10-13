# querying/llm_extractor.py
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

from episcope.utils.extraction_blueprints import DataSourceItem, ExtractionResult, DataSourceSchema


logger = logging.getLogger(__name__)

import ollama

# Helper: locate JSON in a longer model response
def _extract_json_blob(text: str) -> Optional[str]:
    # First attempt: find first { ... } balanced braces substring
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start:i+1]
    # fallback: try to find a substring that looks like JSON using regex (naive)
    m = re.search(r'(\{[\s\S]*\})', text)
    if m:
        return m.group(1)
    return None



from episcope.rag.generation.base import Generator

class LLMExtractor(Generator):
    """Robust LLM wrapper for extracting structured data from chunk text.

    - Accepts an injectable `chat_fn` that performs the chat call:
        chat_fn(model_name: str, messages: list, options: dict) -> {'content': '<str>'}
      This keeps the extractor testable.
    - Validates the model output against DataSourceSchema and returns ExtractionResult.
    """

    def __init__(self, model_name: str = "deepseek-r1:7b", chat_fn: Optional[Callable] = None, enable_eval: bool = False):
        self.model_name = model_name
        self.chat_fn = chat_fn or self._default_ollama_chat
        # optional external evaluator plugin (like codex) can be injected later
        self.enable_eval = enable_eval

    def _default_ollama_chat(self, model_name: str, messages: List[Dict], options: Dict):
        try:
            c = ollama.Client()
            r = c.chat(model=model_name, messages=messages, stream=False, options=options)
            # normalize return
            return {"content": getattr(r, "message", {}).get("content", "") if hasattr(r, 'message') else getattr(r, 'content', '')}
        except Exception as e:
            raise RuntimeError(f"OLLAMA client not available or failed: {e}")

    def _prepare_prompt(self, metadata, chunks_text: str, query: str, paper_type: str) -> str:
        # Single canonical prompt with substitutions; keep it concise
        title = getattr(metadata, "title", "Unknown title")
        abstract = getattr(metadata, "abstract", "") or "Not available"
        prompt = f"""
            Extract structured data sources for this paper.

            Paper type: {paper_type}
            Title: {title}
            Abstract: {abstract}

            Relevant text:
            {chunks_text}

            Question: {query}

            Return a JSON object with keys: data_sources_description, data_sources (list of objects with source_name,url,explanation,section_found), references (optional list).
            {{
                "data_sources_description": "Description of the primary data sources and analysis approach",
                "data_sources": [
                    {{
                        "source_name": "Name of dataset/database/survey",
                        "url": "URL if mentioned or N/A",
                        "explanation": "How this source was used in the analysis",
                        "section_found": "Section where this was mentioned"
                    }}
                ],
                }}
            """
        return prompt.strip()

    def generate(
            self, 
            relevant_chunks: List[Dict], 
            references: List[Any], 
            paper_type: str, 
            metadata: Any, 
            query: str
            ) -> Tuple[ExtractionResult, Optional[Dict]]:
        texts = [c.get("text", "") for c in relevant_chunks]
        combined = "\n\n".join(texts)#[:40_000]  # truncate long text
        prompt = self._prepare_prompt(metadata, combined, query, paper_type)
        messages = [{"role": "user", "content": prompt}]
        print(messages)
        try:
            resp = self.chat_fn(self.model_name, messages, options={"temperature": 0})
            raw = resp.get("content", "")
            # try to directly parse as JSON
            print(raw)
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                # try to find a JSON blob inside text
                blob = _extract_json_blob(raw)
                if blob:
                    parsed = json.loads(blob)
                else:
                    # try python literal
                    import ast
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None

            if parsed is None:
                logger.warning("LLMExtractor: could not parse JSON from model response; returning fallback")
                return ExtractionResult(data_sources_description="PARSE_FAILED", data_sources=[]), None

            # validate with pydantic
            validated = DataSourceSchema(**parsed)
            # convert into ExtractionResult dataclass
            er = ExtractionResult(
                data_sources_description=validated.data_sources_description,
                data_sources=[DataSourceItem(**(ds if isinstance(ds, dict) else {})) for ds in validated.data_sources or []],
                references=validated.references or [],
            )
            return er, None
        except Exception as e:
            logger.exception(f"LLMExtractor failed: {e}")
            return ExtractionResult(data_sources_description="EXTRACTION_ERROR", data_sources=[]), None

