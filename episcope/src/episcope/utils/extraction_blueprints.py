from typing import List, Optional
from pydantic import BaseModel

class ExtractionItemSchema(BaseModel):
    item_type: str
    name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    section_found: Optional[str] = None
    raw_text: Optional[str] = None

class ExtractionResultSchema(BaseModel):
    description: str
    items: List[ExtractionItemSchema] = []
