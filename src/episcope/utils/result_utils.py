import json
from typing import Any, Dict


def create_result_dict(
    paper_id: str,
    pdf_path: str,
    metadata,
    sections,
    references,
    analysis_type: str,
    extraction_result,
    confidence_scores,
) -> Dict[str, Any]:
    references_list = []
    if extraction_result and hasattr(extraction_result, "references"):
        for r in extraction_result.references:
            if hasattr(r, "model_dump"):
                references_list.append(r.model_dump())
            elif isinstance(r, dict):
                references_list.append(r)

    # A safer way to handle 'data_sources'
    data_sources_list = []
    if extraction_result and hasattr(extraction_result, "data_sources"):
        for ds in extraction_result.data_sources:
            if hasattr(ds, "model_dump"):
                data_sources_list.append(ds.model_dump())
            elif isinstance(ds, dict):
                data_sources_list.append(ds)

    return {
        "paper_id": paper_id,
        "paper_title": getattr(metadata, "title", None),
        "first_author": getattr(metadata, "first_author", None),
        "publication_year": getattr(metadata, "publication_year", None),
        "doi": getattr(metadata, "doi", None),
        "journal": getattr(metadata, "journal", None),
        "abstract": (getattr(metadata, "abstract", None) or ""),
        "keywords": json.dumps(getattr(metadata, "keywords", []) or []),
        "analysis_type": analysis_type,
        "num_sections": len(sections),
        "num_references": len(references),
        "data_source_references": sum(
            1 for r in references if getattr(r, "is_data_source", False)
        ),
        "references": json.dumps(references_list),
        "data_sources_description": getattr(
            extraction_result, "data_sources_description", "---"
        )
        if extraction_result
        else "---",
        "data_sources": json.dumps(data_sources_list),
        "pdf_path": pdf_path,
        "confidence_scores": json.dumps(confidence_scores)
        if confidence_scores
        else None,
    }


def create_result_markdown(
    paper_id: str,
    pdf_path: str,
    metadata,
    sections,
    references,
    analysis_type: str,
    extraction_result,
    confidence_scores,
) -> str:
    """Create a markdown summary of the extraction result."""
    md = []
    md.append(f"# Paper ID: {paper_id}\n")
    md.append(f"**Title:** {getattr(metadata, 'title', 'N/A')}\n")
    md.append(f"**First Author:** {getattr(metadata, 'first_author', 'N/A')}\n")
    md.append(f"**Publication Year:** {getattr(metadata, 'publication_year', 'N/A')}\n")
    md.append(f"**DOI:** {getattr(metadata, 'doi', 'N/A')}\n")
    md.append(f"**Journal:** {getattr(metadata, 'journal', 'N/A')}\n")
    md.append(f"**Analysis Type:** {analysis_type}\n")
    md.append(f"**PDF Path:** {pdf_path}\n")
    md.append("\n---\n")

    md.append("## Abstract\n")
    abstract = getattr(metadata, "abstract", "N/A") or "N/A"
    md.append(f"{abstract}\n")

    md.append("## Keywords\n")
    keywords = getattr(metadata, "keywords", []) or []
    if keywords:
        md.append(", ".join(keywords) + "\n")
    else:
        md.append("N/A\n")

    md.append("## Extracted Data Sources\n")
    if extraction_result and hasattr(extraction_result, "data_sources"):
        for ds in extraction_result.data_sources:
            if isinstance(ds, dict):
                source_name = ds.get("source_name", "N/A")
                url = ds.get("url", "N/A")
                explanation = ds.get("explanation", "N/A")
                section_found = ds.get("section_found", "N/A")
            else:  # e.g., DataSource object
                source_name = getattr(ds, "source_name", "N/A")
                url = getattr(ds, "url", "N/A")
                explanation = getattr(ds, "explanation", "N/A")
                section_found = getattr(ds, "section_found", "N/A")
            md.append(f"- **Source Name:** {source_name}\n")
            md.append(f"  - **URL:** {url}\n")
            md.append(f"  - **Explanation:** {explanation}\n")
            md.append(f"  - **Section Found:** {section_found}\n\n")
    else:
        md.append("No data sources extracted.\n")

    md.append("\n---\n")
    md.append("## Data Sources Description\n")
    description = (
        getattr(extraction_result, "data_sources_description", "N/A")
        if extraction_result
        else "N/A"
    )
    md.append(f"{description}\n")

    if confidence_scores:
        md.append("\n---\n")
        md.append("## Confidence Scores\n")
        for key, score in confidence_scores.items():
            md.append(f"- **{key}:** {score}\n\n")

    if extraction_result and hasattr(extraction_result, "matched_dois"):
        md.append("\n---\n")
        md.append("## DOIs\n")
        for doi in extraction_result.matched_dois:
            md.append(f"- https://doi.org/{doi}\n")

    return "\n".join(md)
