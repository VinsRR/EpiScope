from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MatcherConfig:
    """Base configuration for matchers."""
    min_candidate_length: int = 3

@dataclass
class LateInteractionConfig(MatcherConfig):
    """Configuration for the LateInteractionMatcher."""
    transform_model_name: str = "distilbert-base-uncased"
    device: Optional[str] = None
    top_k_prefilter: int = 200
    idf_smoothing: float = 1.0
    max_ref_tokens: int = 256

@dataclass
class SimpleSurnameMatcherConfig(MatcherConfig):
    """Configuration for the SimpleSurnameMatcher."""
    fuzzy_threshold: int = 70
    fuzzy_title_threshold: int = 65

    def __str__(self):
        return "SimpleSurnameMatcherConfig"

@dataclass
class CrossrefConfig:
    """Configuration for the Crossref client."""
    rows: int = 5
    timeout: float = 10.0
    min_title_score: float = 0.85
    user_agent_email: str = "you@example.com"

@dataclass
class OllamaConfig:
    """Configuration for Ollama client interactions."""
    model: str = "qwen2.5vl:3b"
    temperature: float = 0.0
    system_prompt: str = (
        "You are an expert at extracting references from OCR tables. "
        "Respond ONLY with the requested list in plain text, one item per line. "
        "If none, respond with <no references>."
    )
    user_prompt_template: str = (
        "Here is a table:\n\n{csv_data}\n\n"
        "Extract author surnames as short paper references. "
        "Output only the references, one per line. "
        "Return all unique surnames you can find."
    )


import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class GeminiCredentials:
    """Credentials for Gemini."""

    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")






# @dataclass
# class GeminiConfig:
#     """Configuration for Gemini client interactions."""
#     model: str = "gemini-2.5-flash-lite"
#     temperature: float = 0.0
#     system_prompt: str = (
#         "You are an expert at extracting references from scientific papers. "
#         "The user will upload a PDF of a review paper. Your task is to identify and list the references that were "
#         "instrumental for the analysis done in the paper. This is NOT the full list of references, but the specific subset the review is based on. "
#         "Respond ONLY with the requested list in plain text, one item per line. "
#         "If you cannot find the references, provide a short explanation of why (e.g., they are in supplementary materials) "
#         "and if possible, provide a link or instructions on how to find them."
#     )
#     user_prompt: str = (
#         "Please analyze the attached scientific paper and extract the references "
#         "the review is based on"
#     )




from dataclasses import dataclass
from typing import List, Optional, Literal, Type
from pydantic import BaseModel, Field, AnyUrl
import json



class DeclaredIncludedCount(BaseModel):
    value: Optional[int] = Field(
        None, description="Declared number of included studies if stated explicitly."
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            'Short quote or close paraphrase showing the declaration, '
            'e.g., "We included 27 studies", "A total of 14 eligible studies were identified".'
        ),
    )

class SupplementResource(BaseModel):
    label: Optional[str] = Field(
        default=None,
        description='Identifier seen in the paper, e.g., "Supplementary Table S1", "Appendix A", "Online Resource 2", "Additional file 3", "Supplementary Figure 1".'
    )
    href: Optional[AnyUrl] = Field(
        default=None, description="URL to the supplement if provided, ."
    )
    content_note: Optional[str] = Field(
        default=None,
        description='Brief comment on what the supplement contains as stated by the paper, e.g., '
                    '"Full list of included studies", "Data extraction sheet", "Technical Appendix".'
    )


class IncludedStudiesPayload(BaseModel):
    """
    Single JSON object the LLM must output.
    - If included studies are only available in the supplement/appendix and not enumerated in the main paper,
      return studies: [] and populate supplements accordingly.
    - If both in-paper studies and a supplement are present, return both.
    """
    review_type: Optional[str] = None
    declared_included_count: Optional[DeclaredIncludedCount] = None
    inclusion_criteria_summary: Optional[str] = Field(
        default=None, description="1–2 sentences capturing the inclusion scope."
    )
    studies: List[str] = Field(
        default_factory=list,
        description="Full bibliographic citations of included studies. Leave empty if only provided in supplements."
    )
    supplements: Optional[List[SupplementResource]] = Field(
        default=None,
        description="Links/identifiers to supplemental materials that list or describe the included studies."
    )
    # evidence_tables_figures: Optional[List[EvidenceTableFigure]] = Field(
    #     default=None,
    #     description="Tables/figures summarizing included studies (IDs/captions only; no page numbers)."
    # )
    notes: Optional[List[str]] = Field(
        default=None,
        description='Optional notes, e.g., "Included set listed only in Supplementary Table S1", '
                    '"PDF references non-public appendix".'
    )




BASE_SYSTEM_PROMPT = (
    "You are an expert at extracting **included studies** from scientific review papers. "
    "Return **only** the studies explicitly included in the review’s analysis/synthesis—"
    "not the full bibliography and not background-only citations.\n\n"
    "Signals to prioritize for identifying the included set:\n"
    "• PRISMA flow diagram and sections titled “Study selection”, “Included studies”, "
    "  “Characteristics of included studies”.\n"
    "• Explicit declarations of the included count. Typical phrasings include (non-exhaustive):\n"
    "  - \"We included X studies ...\"\n"
    "  - \"A total of X eligible studies were identified ...\"\n"
    "  - \"X studies were included in the review ...\"\n"
    "  - \"The quantitative synthesis comprised X studies ...\"\n"
    "  - \"X studies met the inclusion criteria ...\"\n"
    "• Tables or figures that summarize **included studies** (e.g., “Characteristics of included studies”).\n"
    "• Mentions of **supplementary materials/appendices** containing the included set "
    "(e.g., “Supplementary Table S1”, “Additional file 2”, “Online Resource 3”).\n\n"
    "Supplement handling rules:\n"
    "• If the paper states that the included studies are listed only in a supplement/appendix and does not enumerate"
    " them in the main text, output `studies: []` and populate `supplements` with the link/identifier and a brief "
    " content note reflecting the paper’s wording.\n"
    "• If some or all included studies are listed in the paper **and** supplements are cited, return both: "
    " fill `studies` with the in-paper items and also fill `supplements` (always include supplement links/identifiers "
    " when available, with a short content note).\n"
    "• Populate supplements by extracting both the label and its corresponding URL from the context\n\n"
    "Output format:\n"
    "- Emit exactly one JSON object that **validates** against the JSON Schema below. "
    "  Do not include additional commentary or a second list.\n\n"
    "JSON Schema (the object you must output):\n"
    "{schema_json}\n"
)

BASE_USER_PROMPT = (
    "Analyze the attached review paper and return **only the included studies** used in the review’s analysis/synthesis "
    "(not the full bibliography).\n\n"
    "Please:\n"
    "- Set `review_type`.\n"
    "- Fill `declared_included_count` if the paper states a number (include a short quote/paraphrase in `evidence`).\n"
    "- Provide a 1–2 sentence `inclusion_criteria_summary`.\n"
    "- Populate `studies` with full citations of the included studies. "
    "  If the paper says these are only in a supplement/appendix and does not enumerate them, leave `studies` empty.\n"
    "- Always include any supplement links/identifiers in `supplements`, add a short `content_note` if neccessary for clarity"
    "  (e.g., \"Full list of included studies\").\n"
    "- Use `notes` for anything important (e.g., access limitations).\n\n"
    "Return exactly one JSON object conforming to the provided schema—nothing else."
)

@dataclass
class GeminiConfig:
    """Configuration for Gemini client interactions (schema-injected, supplements-aware, tables/figures-aware)."""
    model: str = "gemini-2.5-pro"  #flash-lite"
    temperature: float = 0.0
    system_prompt: str = ""
    user_prompt: str = ""

    @staticmethod
    def build_prompts(schema_model: Type[BaseModel]) -> "GeminiConfig":
        # Pydantic v2: model_json_schema(); for v1 use .schema()
        schema = schema_model.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        system_prompt = BASE_SYSTEM_PROMPT.format(schema_json=schema_json)
        user_prompt = BASE_USER_PROMPT
        return GeminiConfig(system_prompt=system_prompt, user_prompt=user_prompt)
