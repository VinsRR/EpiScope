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

@dataclass
class GeminiConfig:
    """Configuration for Gemini client interactions."""
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    system_prompt: str = (
        "You are an expert at extracting references from scientific papers. "
        "The user will upload a PDF of a review paper. Your task is to identify and list the references that were "
        "instrumental for the analysis done in the paper. This is NOT the full list of references, but the specific subset the review is based on. "
        "Respond ONLY with the requested list in plain text, one item per line. "
        "If you cannot find the references, provide a short explanation of why (e.g., they are in supplementary materials) "
        "and if possible, provide a link or instructions on how to find them."
    )
    user_prompt: str = (
        "Please analyze the attached scientific paper and extract the references "
        "the review is based on"
    )
