from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .classification import PaperType

@dataclass
class DataSource:
    source_name: str
    url: str = "N/A"
    explanation: str = ""
    section_found: str = ""

@dataclass
class ExtractionItem:
    item_type: str
    name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    section_found: Optional[str] = None
    raw_text: Optional[str] = None

@dataclass
class ExtractionResult():
    description: str
    items: List[ExtractionItem] = field(default_factory=list)

@dataclass
class ClassificationResult:
    paper_type: 'PaperType'
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)

@dataclass
class SearchResult:
    """Represents a search result chunk with metadata."""
    id: str
    paper_id: str 
    text: str
    section_type: str = "other"
    title: str = ""
    similarity_score: float = 0.0
    source: str = ""  # semantic, keyword, context
    artifacts: Dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0
