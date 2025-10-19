import uuid
from typing import List, Dict, Any
from episcope.schemas import StructuredSection, PaperMetadata
from episcope.utils.chunking import paragraph_chunking

def chunk_paper(
    sections: List[StructuredSection],
    metadata: PaperMetadata,
    paper_id: str,
    min_chunk_size: int = 50,
) -> List[Dict[str, Any]]:
    """Creates a list of chunks from paper sections and metadata."""
    chunks: List[Dict[str, Any]] = []
    if metadata and metadata.abstract:
        text = metadata.abstract
        chunks.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
            "text": text,
            "content": text,
            "section_title": "Abstract",
            "section_type": "Abstract",
            "paper_id": paper_id,
            "is_metadata": True,
        })

    for section in sections:
        section_chunks = paragraph_chunking(section, min_chunk_size)
        for chunk in section_chunks:
            text = chunk["text"]
            chunk_meta = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
                "text": text,
                "content": text,
                "section_title": section.title,
                "section_type": section.section_type,
                "paper_id": paper_id,
                "is_metadata": False,
            }
            chunks.append(chunk_meta)
    return chunks
