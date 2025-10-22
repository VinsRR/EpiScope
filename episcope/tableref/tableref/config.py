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
