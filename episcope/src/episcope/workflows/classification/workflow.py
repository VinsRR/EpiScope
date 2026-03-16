import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ValidationError

from episcope.db.academic_db import AcademicDB
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.schemas import PaperMetadata
from episcope.workflows.base import AbstractRAG
from episcope.workflows.classification.config import BaseClassifierConfig, PaperTypeClassifierConfig

from .schemas import ClassificationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationParseError(Exception):
    """Raised when an LLM response cannot be parsed into a ClassificationResult."""


class PaperClassifier(AbstractRAG):
    """Multi-modal paper classification using RAG and semantic similarity."""

    def __init__(
        self,
        retriever: AbstractRetriever,
        generator: Generator,
        strategy_name: str = None,
        config: Optional[BaseClassifierConfig] = None,
        academic_db: Optional[AcademicDB] = None,
    ):
        super().__init__(retriever, generator)
        self.config = config or PaperTypeClassifierConfig()
        self.academic_db = academic_db
        self.strategy_name = strategy_name

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(self, paper_id: str, metadata: Optional[PaperMetadata] = None) -> ClassificationResult:
        """Run the classification workflow for a single paper."""
        if self.academic_db:
            metadata = self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        elif not metadata:
            raise ValueError("metadata must be provided when academic_db is not available.")

        relevant_chunks = self.get_relevant_chunks(metadata, paper_id, top_k=self.config.top_k)
        return self.classify_based_on_relevant_chunks(relevant_chunks, metadata)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def get_relevant_chunks(
        self, metadata: PaperMetadata, paper_id: str, top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve and rank relevant chunks for each paper type."""
        aggregated_chunks: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for paper_type, templates in self.config.template_paragraphs.items():
            for template_query in templates:
                retrieved = self.retriever.retrieve_by_paper(template_query, paper_id, top_k=top_k)
                for chunk in retrieved:
                    aggregated_chunks[paper_type].append((chunk.text, chunk.similarity_score))

        return self._deduplicate_and_rank_chunks(aggregated_chunks, top_k)

    def _deduplicate_and_rank_chunks(
        self,
        aggregated_chunks: Dict[str, List[Tuple[str, float]]],
        top_k: int,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Remove duplicates and keep the top-k highest-scoring chunks per paper type."""
        final_chunks = {}
        for paper_type, chunks in aggregated_chunks.items():
            unique: Dict[str, float] = {}
            for text, score in chunks:
                text = text.strip()
                if text and score > unique.get(text, float("-inf")):
                    unique[text] = score
            final_chunks[paper_type] = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return final_chunks

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def classify_based_on_relevant_chunks(
        self,
        relevant_chunks: Dict[str, List[Tuple[str, float]]],
        metadata: PaperMetadata,
    ) -> ClassificationResult:
        if not any(relevant_chunks.values()):
            logger.warning("No relevant chunks found for classification.")
        return self._llm_classify(metadata, relevant_chunks)

    def _llm_classify(
        self,
        metadata: PaperMetadata,
        relevant_chunks: Dict[str, List[Tuple[str, float]]],
    ) -> ClassificationResult:
        """LLM-based classification with self-correcting retries.

        On each attempt:
          1. Call the LLM.
          2. Try to parse the response.
          3. If parsing fails, append the error as a corrective user turn and retry.
        """
        messages = self._build_initial_prompt(metadata, relevant_chunks)
        last_response: Optional[str] = None

        for attempt in range(1, self.config.max_validation_retries + 1):
            # --- Generation ---
            try:
                provenance = self.generator.generate(
                    contexts=list(relevant_chunks.values()),
                    message_builder=lambda **kwargs: messages,
                    format="json",
                )
                last_response = provenance.answer
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt} failed: {e}")
                continue

            # --- Parsing ---
            try:
                return self._parse_classification_response(last_response)
            except ClassificationParseError as e:
                logger.warning(f"Parse attempt {attempt}/{self.config.max_validation_retries} failed: {e}")
                logger.debug(f"Raw response that failed parsing:\n{last_response}")
                if attempt < self.config.max_validation_retries:
                    # Feed the error back so the LLM can self-correct.
                    messages.append({"role": "assistant", "content": last_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid. Error: {e}\n"
                            "Please return ONLY a single valid JSON object that matches the schema. "
                            "Do not include any extra text, markdown fences, or explanation."
                        ),
                    })

        # All attempts exhausted.
        logger.error(
            f"Classification failed after {self.config.max_validation_retries} attempt(s). "
            "Returning fallback result."
        )
        if last_response is not None:
            logger.error(f"Last raw response:\n{last_response}")
        return self._create_fallback_result()

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _format_chunks_for_prompt(self, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> str:
        chunks_info = ""
        for paper_type, chunks in relevant_chunks.items():
            if not chunks:
                continue
            avg_score = np.mean([score for _, score in chunks])
            chunks_info += f"\n{paper_type.upper()} (avg {avg_score:.3f}):\n"
            for text, score in chunks:
                chunks_info += f"  • {score:.3f}: {text}\n"
        return chunks_info

    def _build_initial_prompt(
        self,
        metadata: PaperMetadata,
        relevant_chunks: Dict[str, List[Tuple[str, float]]],
    ) -> List[Dict[str, str]]:
        chunks_info = self._format_chunks_for_prompt(relevant_chunks)
        schema = self.config.output_schema.model_json_schema()
        categories = "\n".join(f"{key} – {value}" for key, value in self.config.category_labels.items())

        user_prompt = self.config.user_prompt_template.format(
            categories=categories,
            title=metadata.title,
            abstract=metadata.abstract or "N/A",
            keywords=", ".join(metadata.keywords or []),
            chunks_info=chunks_info,
            schema=json.dumps(schema),
            **self.config.extra_output_fields,
        )
        system_prompt = self.config.system_prompt.format(
            n_categories=len(self.config.category_labels),
            category_labels=", ".join(self.config.category_labels.values()),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # -------------------------------------------------------------------------
    # Response parsing  (raises ClassificationParseError on any failure)
    # -------------------------------------------------------------------------

    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        """Parse an LLM response into a ClassificationResult.

        Supports raw JSON, ```json fenced blocks, and mixed text+JSON.
        Raises ClassificationParseError on any failure so the caller can retry.
        """
        raw = response_content.strip()

        parsed = self._parse_json(raw)          # raises ClassificationParseError on failure
        classification = self._extract_classification(parsed)  # raises on failure
        extras = self._extract_extras(parsed)

        return ClassificationResult(
            classification=classification,
            confidence=float(parsed.confidence) if getattr(parsed, "confidence", None) is not None else 0.0,
            class_probabilities=parsed.class_probabilities or {},
            evidence={"reasoning": parsed.reasoning},
            extras=extras,
        )

    # -- JSON extraction helpers ----------------------------------------------

    def _parse_json(self, raw: str) -> Any:
        """Try to extract and validate a JSON object from `raw`.

        Fast path: validate the whole string. Slow path: find the first balanced
        JSON object (preferring fenced blocks).
        Raises ClassificationParseError if nothing valid is found.
        """
        # Fast path
        try:
            return self.config.output_schema.model_validate_json(raw)
        except Exception:
            pass

        candidate = self._extract_json_candidate(raw)
        if candidate is None:
            raise ClassificationParseError("No JSON object found in model output.")

        try:
            json.loads(candidate)   # syntax check before Pydantic
        except json.JSONDecodeError as e:
            raise ClassificationParseError(f"Extracted JSON is not valid JSON: {e}") from e

        try:
            return self.config.output_schema.model_validate_json(candidate)
        except ValidationError as e:
            raise ClassificationParseError(f"JSON does not match schema: {e}") from e
        except Exception as e:
            raise ClassificationParseError(f"Unexpected validation error: {e}") from e

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[str]:
        """Return the first JSON object from `text`, preferring fenced blocks."""
        # 1) Fenced blocks (```json ... ``` or ``` ... ```)
        for pat in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
            m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # 2) First balanced brace pair
        start = text.find("{")
        if start == -1:
            return None

        depth, in_string, escape = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_string:
                # NOTE: evaluate old `escape` before updating it.
                if escape:
                    escape = False          # this char was escaped; clear flag
                elif ch == "\\":
                    escape = True           # next char is escaped
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()

        return None

    # -- Classification / extras extraction -----------------------------------

    def _extract_classification(self, parsed: Any) -> List[Any]:
        """Pull the classification label(s) out of a parsed schema object.

        Raises ClassificationParseError if the object has neither a
        'classification' nor a 'primary_label' attribute.
        """
        if hasattr(parsed, "classification"):
            raw_cls = parsed.classification
            if isinstance(raw_cls, list):
                mapped = [
                    self.config.classification_mapping[code]
                    for code in raw_cls
                    if code in self.config.classification_mapping
                ]
                return mapped or self.config.default_classification
            else:
                mapped = self.config.classification_mapping.get(raw_cls)
                return [mapped] if mapped is not None else self.config.default_classification

        if hasattr(parsed, "primary_label"):
            return [parsed.primary_label]

        raise ClassificationParseError(
            "Parsed output has no 'classification' or 'primary_label' attribute."
        )

    @staticmethod
    def _extract_extras(parsed: Any) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        if hasattr(parsed, "extras") and parsed.extras:
            extras = (
                parsed.extras.model_dump()
                if hasattr(parsed.extras, "model_dump")
                else dict(parsed.extras)
            )
        if hasattr(parsed, "secondary_labels") and parsed.secondary_labels:
            extras["secondary_labels"] = parsed.secondary_labels
        return extras

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    def _create_fallback_result(self) -> ClassificationResult:
        n = len(self.config.category_labels)
        uniform_prob = 1 / n if n else 0.0
        return ClassificationResult(
            classification=self.config.default_classification,
            confidence=0.0,
            class_probabilities={
                label: uniform_prob for label in self.config.category_labels.values()
            },
            evidence={"reasoning": "Classification failed, defaulting to unclear"},
            extras={},
        )


# import json
# import logging
# import re
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple, Any
# import numpy as np
# from episcope.db.academic_db import AcademicDB
# from episcope.workflows.classification.config import BaseClassifierConfig, PaperTypeClassifierConfig
# from episcope.workflows.base import AbstractRAG
# from episcope.rag.generation.base import Generator
# from episcope.rag.interfaces import AbstractRetriever
# from episcope.schemas import PaperMetadata
# from .schemas import ClassificationResult
# from pydantic import ValidationError




# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class PaperClassifier(AbstractRAG):
#     """Multi-modal paper classification using RAG and semantic similarity."""

#     def __init__(self,
#                  retriever: AbstractRetriever,
#                  generator: Generator,
#                  strategy_name: str = None,
#                  config: Optional[BaseClassifierConfig] = None,
#                  academic_db: Optional[AcademicDB] = None
#                  ):
#         super().__init__(retriever, generator)
#         self.config = config or PaperTypeClassifierConfig()
#         self.academic_db = academic_db
#         self.strategy_name = strategy_name

#     def run(self, paper_id: str, metadata: Optional[PaperMetadata] = None) -> ClassificationResult:
#         """
#         Run the classification workflow for a single paper.
#         """
#         if self.academic_db:
#             metadata = self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
#         elif not metadata:
#             raise ValueError("metadata must be provided when academic_db is not available.")

#         relevant_chunks = self.get_relevant_chunks(metadata, paper_id, top_k=self.config.top_k)
#         return self.classify_based_on_relevant_chunks(relevant_chunks, metadata)

#     def get_relevant_chunks(self, metadata: PaperMetadata, paper_id: str, top_k: int = 10) -> Dict[
#         str, List[Tuple[str, float]]]:
#         """Main method to retrieve relevant chunks for each paper type."""
#         aggregated_chunks = defaultdict(list)
#         for paper_type, templates in self.config.template_paragraphs.items():
#             for template_query in templates:
#                 retrieved_chunks = self.retriever.retrieve_by_paper(template_query, paper_id, top_k=top_k)
#                 for chunk in retrieved_chunks:
#                     aggregated_chunks[paper_type].append((chunk.text, chunk.similarity_score))

#         return self._deduplicate_and_rank_chunks(aggregated_chunks, top_k)

#     def _deduplicate_and_rank_chunks(self, aggregated_chunks: Dict[str, List[Tuple[str, float]]],
#                                      top_k: int) -> Dict[str, List[Tuple[str, float]]]:
#         """Remove duplicates and keep top-k chunks for each paper type."""
#         final_chunks = {}
#         for paper_type, chunks in aggregated_chunks.items():
#             unique_chunks = {}
#             for text, score in chunks:
#                 text = text.strip()
#                 if text and (text not in unique_chunks or score > unique_chunks[text]):
#                     unique_chunks[text] = score

#             sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1], reverse=True)[:top_k]
#             final_chunks[paper_type] = sorted_chunks
#         return final_chunks

#     def classify_based_on_relevant_chunks(self, relevant_chunks: Dict[str, List[Tuple[str, float]]],
#                                           metadata: PaperMetadata) -> ClassificationResult:
#         """Classify paper based on relevant chunks using LLM."""
#         if not any(relevant_chunks.values()):
#             logger.warning("No relevant chunks found for classification.")

#         return self._llm_classify(metadata, relevant_chunks)

#     def _llm_classify(self, metadata: PaperMetadata,
#                       relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> ClassificationResult:
#         """Perform LLM-based classification with self-correction."""
#         messages = self._build_initial_prompt(metadata, relevant_chunks)


#         for attempt in range(self.config.max_validation_retries):
#             try:
#                 provenance = self.generator.generate(
#                     contexts=list(relevant_chunks.values()),
#                     message_builder=lambda **kwargs: messages,
#                     format="json"
#                 )
#                 response_content = provenance.answer

#                 # print("\n\nmessages:", messages)
#                 # print("\n\nresponse_content:", response_content, "\n\n")

#             except Exception as e:
#                 logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
#                 continue

#             try:
#                 return self._parse_classification_response(response_content)
#             except Exception as e:
#                 logger.warning(f"Classification parsing attempt {attempt + 1} failed: {e}")
#                 error_message = f"The JSON output is invalid. Please fix it. Error: {e}"
#                 messages.append({"role": "assistant", "content": response_content})
#                 messages.append({"role": "user", "content": error_message})
#                 print("\n\nresponse_content:", response_content, "\n\n")
#         print("\n\nresponse_content:", response_content, "\n\n")
#         return self._create_fallback_result()

#     def _format_chunks_for_prompt(self, relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> str:
#         """Format chunks information for the LLM prompt."""
#         chunks_info = ""
#         for paper_type, chunks in relevant_chunks.items():
#             if not chunks:
#                 continue
#             avg_score = np.mean([score for _, score in chunks]) if chunks else 0
#             chunks_info += f"\n{paper_type.upper()} (avg {avg_score:.3f}):\n"
#             for text, score in chunks:
#                 chunks_info += f"  • {score:.3f}: {text}\n"
#         return chunks_info

#     def _build_initial_prompt(self, metadata: PaperMetadata,
#                               relevant_chunks: Dict[str, List[Tuple[str, float]]]) -> List[Dict[str, str]]:
#         """Create the initial classification prompt."""
#         chunks_info = self._format_chunks_for_prompt(relevant_chunks)
#         schema = self.config.output_schema.model_json_schema()
#         categories = "\n".join(f"{key} – {value}" for key, value in self.config.category_labels.items())

#         prompt_args = {
#             'categories': categories,
#             'title': metadata.title,
#             'abstract': metadata.abstract or 'N/A',
#             'keywords': ', '.join(metadata.keywords or []),
#             'chunks_info': chunks_info,
#             'schema': json.dumps(schema),
#             **self.config.extra_output_fields
#         }
#         user_prompt = self.config.user_prompt_template.format(**prompt_args)
#         system_prompt = self.config.system_prompt.format(
#             n_categories=len(self.config.category_labels),
#             category_labels=", ".join(list(self.config.category_labels.values()))
#         )
#         return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ]





#     def _parse_classification_response(self, response_content: str) -> ClassificationResult:
#         """Parse LLM response into ClassificationResult.

#         Supports raw JSON, ```json fenced blocks, and mixed text+JSON (extracts first JSON object).
#         """
#         def _extract_json_candidate(text: str) -> Optional[str]:
#             """Extract a JSON object from text, preferring fenced blocks."""
#             # Prefer fenced ```json blocks
#             for pat in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
#                 m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
#                 if m:
#                     return m.group(1).strip()

#             # Fallback: extract first balanced JSON object via brace matching
#             start = text.find("{")
#             if start == -1:
#                 return None

#             depth, in_string, escape = 0, False, False
#             for i, ch in enumerate(text[start:], start):
#                 if in_string:
#                     escape = ch == "\\" and not escape
#                     if not escape and ch == '"':
#                         in_string = False
#                 elif ch == '"':
#                     in_string = True
#                 elif ch == "{":
#                     depth += 1
#                 elif ch == "}" :
#                     depth -= 1
#                     if depth == 0:
#                         return text[start:i + 1].strip()

#             return None

#         def _log_and_fallback(reason: str) -> ClassificationResult:
#             logger.error(f"{reason} Raw response:\n{raw}")
#             return self._create_fallback_result()

#         def _validate_candidate(candidate: str) -> Optional[Any]:
#             """Validate a JSON string against the output schema. Returns parsed model or None."""
#             try:
#                 json.loads(candidate)  # Sanity-check before handing to pydantic
#             except json.JSONDecodeError:
#                 return None
#             try:
#                 return self.config.output_schema.model_validate_json(candidate)
#             except (ValidationError, Exception) as e:
#                 logger.error(f"Schema validation failed: {e}")
#                 return None

#         # --- Parse ---
#         raw = response_content.strip()

#         parsed = None
#         # Fast path: try raw content directly
#         try:
#             parsed = self.config.output_schema.model_validate_json(raw)
#         except Exception:
#             candidate = _extract_json_candidate(raw)
#             if candidate is None:
#                 return _log_and_fallback("Could not find any JSON object in model output.")
#             parsed = _validate_candidate(candidate)
#             if parsed is None:
#                 return _log_and_fallback("Extracted JSON candidate failed validation.")

#         # --- Extract classification ---
#         if hasattr(parsed, "classification"):
#             raw_cls = parsed.classification
#             if isinstance(raw_cls, list):
#                 classification = [
#                     self.config.classification_mapping[code]
#                     for code in raw_cls
#                     if code in self.config.classification_mapping
#                 ] or self.config.default_classification
#             else:
#                 mapped = self.config.classification_mapping.get(raw_cls)
#                 classification = [mapped] if mapped is not None else self.config.default_classification

#         elif hasattr(parsed, "primary_label"):
#             classification = [parsed.primary_label]
#         else:
#             return _log_and_fallback(
#                 "Parsed output has no 'classification' or 'primary_label' attribute."
#             )

#         # --- Extract extras ---
#         extras = {}
#         if hasattr(parsed, "extras") and parsed.extras:
#             extras = parsed.extras.model_dump() if hasattr(parsed.extras, "model_dump") else parsed.extras
#         if hasattr(parsed, "secondary_labels") and parsed.secondary_labels:
#             extras["secondary_labels"] = parsed.secondary_labels

#         return ClassificationResult(
#             classification=classification,
#             confidence=float(parsed.confidence) if getattr(parsed, "confidence", None) is not None else 0.0,
#             class_probabilities=parsed.class_probabilities or {},
#             evidence={"reasoning": parsed.reasoning},
#             extras=extras,
#         )


#     def _create_fallback_result(self) -> ClassificationResult:
#         """Create a fallback classification result when LLM fails."""
#         return ClassificationResult(
#             classification=self.config.default_classification,
#             confidence=0.0,
#             class_probabilities={label: 1 / len(self.config.category_labels) for label in
#                                  self.config.category_labels.values() if self.config.category_labels},
#             evidence={"reasoning": "Classification failed, defaulting to unclear"},
#             extras={}
#         )










    # def _parse_classification_response(self, response_content: str) -> ClassificationResult:
    #     """Parse LLM response into ClassificationResult."""
    #     if response_content.startswith("```json"):
    #         response_content = (
    #             response_content
    #             .replace("```json", "")
    #             .replace("```", "")
    #             .strip()
    #         )

    #     parsed = self.config.output_schema.model_validate_json(response_content)
    #     extras = {}
    #     if hasattr(parsed, "extras") and parsed.extras:
    #         if hasattr(parsed.extras, "model_dump"):
    #             extras = parsed.extras.model_dump()
    #         else:
    #             extras = parsed.extras

    #     if hasattr(parsed, 'classification'):
    #         raw_cls = parsed.classification
    #         default_cls = self.config.default_classification

    #         if isinstance(raw_cls, list):
    #             mapped: list = []
    #             for code in raw_cls:
    #                 mapped_value = self.config.classification_mapping.get(code)
    #                 if mapped_value is not None:
    #                     mapped.append(mapped_value)

    #             if not mapped:
    #                 mapped = default_cls

    #             classification = mapped

    #         else:
    #             # Backward compatibility: classification is a single letter
    #             mapped_value = self.config.classification_mapping.get(raw_cls)
    #             if mapped_value is not None:
    #                 classification = [mapped_value]
    #             else:
    #                 classification = default_cls
    #     elif hasattr(parsed, 'primary_label'):  # Special case for PaperTypeClassificationOutput
    #         classification = [parsed.primary_label]
    #         if hasattr(parsed, 'secondary_labels') and parsed.secondary_labels:
    #             extras['secondary_labels'] = parsed.secondary_labels
    #     else:
    #         # Fallback or error
    #         logger.error("Parsed classification output has no 'classification' or 'primary_label' attribute.")
    #         return self._create_fallback_result()

    #     confidence = float(parsed.confidence) if parsed.confidence is not None else 0.0
    #     class_probs = parsed.class_probabilities or {}

    #     return ClassificationResult(
    #         classification=classification,
    #         confidence=confidence,
    #         class_probabilities=class_probs,
    #         evidence={"reasoning": parsed.reasoning},
    #         extras=extras
    #     )




    # def _parse_classification_response(self, response_content: str) -> ClassificationResult:
    #     """Parse LLM response into ClassificationResult.

    #     Supports:
    #     - raw JSON
    #     - ```json fenced blocks anywhere in the text
    #     - text + JSON (extract first JSON object)
    #     """
    #     def _extract_json_candidate(text: str) -> Optional[str]:
    #         # 1) Prefer fenced ```json blocks (anywhere)
    #         fence_patterns = [
    #             r"```json\s*(\{.*?\})\s*```",   # explicit json fence
    #             r"```\s*(\{.*?\})\s*```",       # any fence containing a JSON object
    #         ]
    #         for pat in fence_patterns:
    #             m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
    #             if m:
    #                 return m.group(1).strip()

    #         # 2) Fallback: extract first balanced JSON object via brace matching
    #         start = text.find("{")
    #         if start == -1:
    #             return None

    #         depth = 0
    #         in_string = False
    #         escape = False
    #         for i in range(start, len(text)):
    #             ch = text[i]

    #             if in_string:
    #                 if escape:
    #                     escape = False
    #                 elif ch == "\\":
    #                     escape = True
    #                 elif ch == '"':
    #                     in_string = False
    #                 continue

    #             if ch == '"':
    #                 in_string = True
    #                 continue

    #             if ch == "{":
    #                 depth += 1
    #             elif ch == "}":
    #                 depth -= 1
    #                 if depth == 0:
    #                     return text[start : i + 1].strip()

    #         return None

    #     # --- Attempt parsing ---
    #     raw = response_content.strip()

    #     # Keep your old "startswith fence" cleanup for backward compatibility,
    #     # but it won't be sufficient alone (the fence might not be at the start).
    #     if raw.startswith("```json"):
    #         raw = raw.replace("```json", "").replace("```", "").strip()

    #     parsed = None
    #     json_payload = None

    #     # 1) First: try direct JSON validation (fast path)
    #     try:
    #         parsed = self.config.output_schema.model_validate_json(raw)
    #         json_payload = raw
    #     except Exception:
    #         # 2) Try extracting a JSON candidate and validating that
    #         candidate = _extract_json_candidate(raw)
    #         if candidate is None:
    #             logger.error("Could not find any JSON object in model output.")
    #             return self._create_fallback_result()

    #         # Optionally sanity-check candidate is JSON before giving it to pydantic
    #         try:
    #             json.loads(candidate)
    #         except json.JSONDecodeError:
    #             logger.error("Extracted JSON candidate is not valid JSON.")
    #             return self._create_fallback_result()

    #         try:
    #             parsed = self.config.output_schema.model_validate_json(candidate)
    #             json_payload = candidate
    #         except ValidationError as e:
    #             logger.error(f"Schema validation failed for extracted JSON: {e}")
    #             return self._create_fallback_result()
    #         except Exception as e:
    #             logger.error(f"Unexpected error validating extracted JSON: {e}")
    #             return self._create_fallback_result()

    #     # --- Extras handling (unchanged, but guarded) ---
    #     extras = {}
    #     if hasattr(parsed, "extras") and parsed.extras:
    #         if hasattr(parsed.extras, "model_dump"):
    #             extras = parsed.extras.model_dump()
    #         else:
    #             extras = parsed.extras

    #     # --- Classification extraction logic 
    #     if hasattr(parsed, "classification"):
    #         raw_cls = parsed.classification
    #         default_cls = self.config.default_classification

    #         if isinstance(raw_cls, list):
    #             mapped: list = []
    #             for code in raw_cls:
    #                 mapped_value = self.config.classification_mapping.get(code)
    #                 if mapped_value is not None:
    #                     mapped.append(mapped_value)

    #             if not mapped:
    #                 mapped = default_cls

    #             classification = mapped
    #         else:
    #             # Backward compatibility: classification is a single code
    #             mapped_value = self.config.classification_mapping.get(raw_cls)
    #             if mapped_value is not None:
    #                 classification = [mapped_value]
    #             else:
    #                 classification = default_cls

    #     elif hasattr(parsed, "primary_label"):  # Special case for PaperTypeClassificationOutput
    #         classification = [parsed.primary_label]
    #         if hasattr(parsed, "secondary_labels") and parsed.secondary_labels:
    #             extras["secondary_labels"] = parsed.secondary_labels
    #     else:
    #         logger.error(
    #             "Parsed classification output has no 'classification' or 'primary_label' attribute."
    #         )
    #         return self._create_fallback_result()

    #     confidence = float(parsed.confidence) if getattr(parsed, "confidence", None) is not None else 0.0
    #     class_probs = parsed.class_probabilities or {}

    #     return ClassificationResult(
    #         classification=classification,
    #         confidence=confidence,
    #         class_probabilities=class_probs,
    #         evidence={"reasoning": parsed.reasoning},
    #         extras=extras,
    #     )



