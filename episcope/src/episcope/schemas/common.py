from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Chunk:
    id: str
    section_type: str
    title: str
    text: str
    metadata: Dict = field(default_factory=dict)
