from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class SearchResult:
    """Represents a search result chunk with metadata."""
    id: str
    paper_id: str
    text: str
    section_type: str = "other"
    section_title: str = ""
    title: str = ""
    is_metadata: bool = False
    similarity_score: float = 0.0
    source: str = ""  # semantic, keyword, context
    artifacts: Dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0
