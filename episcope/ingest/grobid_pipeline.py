"""High level GROBID extraction pipeline for PrecisionMiner.

This module glues together the document loading facilities
(:class:`episcope.ingest.document_loader.GrobidDocumentLoader`) and
the persistence layer (:class:`episcope.storage.academic_db_manager.AcademicDBManager`) to
produce fully structured extraction outputs from raw PDF files.  The
primary entry point, :func:`extract_paper`, accepts a file path and
a strategy name, runs GROBID to obtain sections, metadata and
references, stores each component into the database with the
appropriate compound key, and optionally writes the JSON files
to the local filesystem for debugging or offline inspection.

The functions in this module deliberately avoid any heavy
dependencies on LLMs or embedding libraries; they focus solely on
parsing and persistence.  Downstream RAG components can query
the database to obtain inputs for indexing, retrieval and
generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .document_loader import DocumentLoaderFactory, GrobidDocumentLoader
from ..storage.academic_db_manager import AcademicDBManager

logger = logging.getLogger(__name__)


def extract_paper(
    file_path: str | Path,
    *,
    strategy_name: str,
    db: AcademicDBManager,
    grobid_url: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Extract structured data from a single PDF and persist it.

    This function uses :class:`GrobidDocumentLoader` to obtain
    sections, metadata and references from a PDF file.  The results
    are inserted into the provided :class:`AcademicDBManager` under
    the specified ``strategy_name``.  Optionally, the same data are
    written to disk in a hierarchical directory structure for
    debugging.  Existing entries for the same (paper_id, data_type,
    strategy_name) are not overwritten.

    Args:
        file_path: Location of the PDF file to process.
        strategy_name: Namespace under which to store the results.
        db: Instance of :class:`AcademicDBManager` to persist data.
        grobid_url: Optional custom URL for the GROBID service.
        output_dir: Optional base directory for writing local JSON
            files.  If omitted or ``None`` local writing is skipped.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    paper_id = path.stem
    # Instantiate loader with custom server URL if provided
    loader: GrobidDocumentLoader = DocumentLoaderFactory.get_loader(
        "grobid", grobid_url=grobid_url
    )  # type: ignore[assignment]
    sections, metadata, references = loader.load_with_references(path)
    # Convert to plain dicts for storage
    sections_dicts: List[Dict] = [s.to_dict() for s in sections]
    metadata_dict: Dict = metadata.to_dict()
    references_dicts: List[Dict] = [r.to_dict() for r in references]
    # Persist each component
    db.insert(paper_id, "sections", strategy_name, sections_dicts)
    db.insert(paper_id, "metadata", strategy_name, metadata_dict)
    db.insert(paper_id, "references", strategy_name, references_dicts)
    logger.info(
        f"Persisted extraction for {paper_id} under strategy {strategy_name}."
    )
    # Optionally write out JSON files locally
    if output_dir:
        _write_local_jsons(
            paper_id,
            sections_dicts,
            metadata_dict,
            references_dicts,
            strategy_name,
            base_dir=Path(output_dir),
        )


def extract_directory(
    dir_path: str | Path,
    *,
    strategy_name: str,
    db: AcademicDBManager,
    grobid_url: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Process all PDF files in a directory.

    Recursively walks the directory tree rooted at ``dir_path`` and
    applies :func:`extract_paper` to each ``*.pdf`` file.  Non‑PDF
    files are ignored.  Errors are logged and skipped rather than
    aborting the entire run.  Local file writing is controlled by
    ``output_dir``.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise ValueError(f"Expected directory: {dir_path}")
    for pdf in root.rglob("*.pdf"):
        try:
            extract_paper(
                pdf,
                strategy_name=strategy_name,
                db=db,
                grobid_url=grobid_url,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.error(f"Failed to process {pdf}: {exc}")


def _write_local_jsons(
    paper_id: str,
    sections: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    references: List[Dict[str, Any]],
    strategy_name: str,
    *,
    base_dir: Path,
) -> None:
    """Write extracted JSON objects to disk for debugging.

    The files are written under ``base_dir`` organised by
    ``strategy_name`` and ``paper_id``.  For example::

        /base_dir/Strategy_V1_GROBID_Standard/Paper123/sections.json
        /base_dir/Strategy_V1_GROBID_Standard/Paper123/metadata.json
        /base_dir/Strategy_V1_GROBID_Standard/Paper123/references.json

    If the directories do not exist they are created.  Existing
    files are overwritten.
    """
    paper_dir = base_dir / strategy_name / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sections.json": {"items": sections},
        "metadata.json": metadata,
        "references.json": {"items": references},
    }
    for name, obj in files.items():
        out_path = paper_dir / name
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, ensure_ascii=False)
            logger.debug(f"Wrote {out_path}")
        except Exception as exc:
            logger.warning(f"Failed to write {out_path}: {exc}")




        


# class GrobidClient:
#     """ GROBID client that fully utilizes structured output"""
    
#     def __init__(self, grobid_server: str = "http://localhost:8070", timeout: int = 3600):
#         self.grobid_server = grobid_server
#         self.timeout = timeout  
#         try:
#             from grobid_client.grobid_client import GrobidClient as PyClient
#             self.client = PyClient(grobid_server=grobid_server, timeout=timeout)
#             logger.info(f" GROBID client initialized with server: {grobid_server}")
#         except ImportError:
#             logger.error("grobid_client_python not installed. Install with: pip install grobid_client_python")
#             raise

#     def process_fulltext(self, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
#         """Process full document with GROBID and extract all structured information"""
#         try:
#             # Process the full document
#             result = self.client.process_pdf(
#                 service="processFulltextDocument",
#                 pdf_file=pdf_path,
#                 generateIDs=True,
#                 consolidate_header=True, 
#                 consolidate_citations=True, 
#                 include_raw_citations=True, 
#                 include_raw_affiliations=True, 
#                 tei_coordinates=True, 
#                 segment_sentences=True
#             )
            
#             # Handle different return formats
#             if isinstance(result, tuple) and len(result) >= 3:
#                 xml_content = result[2]
#             elif isinstance(result, str):
#                 xml_content = result
#             else:
#                 logger.warning(f"Unexpected result format from GROBID: {type(result)}")
#                 return self._create_fallback_data(pdf_path)

#             if not xml_content or not xml_content.strip().startswith('<'):
#                 logger.warning(f"No valid XML returned for {pdf_path}")
#                 return self._create_fallback_data(pdf_path)

#             return self._parse_fulltext_tei(xml_content, pdf_path)

#         except Exception as e:
#             logger.error(f"GROBID fulltext processing failed for {pdf_path}: {e}")
#             return self._create_fallback_data(pdf_path)

#     def _parse_fulltext_tei(self, xml_content: str, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
#         """Parse full TEI XML to extract comprehensive structured information"""
#         try:
#             root = etree.fromstring(xml_content.encode())
            
#             # Handle namespaces
#             nsmap = root.nsmap
#             tei_ns = nsmap.get(None) or 'http://www.tei-c.org/ns/1.0'
#             ns = {'tei': tei_ns}

#             # Extract metadata
#             metadata = self._extract__metadata(root, ns, pdf_path)
            
#             # Extract structured sections
#             sections = self._extract_structured_sections(root, ns)
            
#             # Extract references
#             references = self._extract_references(root, ns)
            
#             logger.info(f"Extracted {len(sections)} sections and {len(references)} references from {pdf_path}")
#             return metadata, sections, references

#         except Exception as e:
#             logger.error(f"Failed to parse fulltext TEI XML for {pdf_path}: {e}")
#             return self._create_fallback_data(pdf_path)

#     def _extract__metadata(self, root, ns: Dict[str, str], pdf_path: str) -> PaperMetadata:
#         """Extract comprehensive metadata from TEI header"""
#         # Extract title
#         title_elem = root.find('.//tei:title[@level="a"]', namespaces=ns)
#         if title_elem is None:
#             title_elem = root.find('.//tei:title', namespaces=ns)
#         title = self._get_element_text(title_elem) or Path(pdf_path).stem

#         # Extract authors
#         authors = []
#         for person in root.findall('.//tei:author//tei:persName', namespaces=ns):
#             forename = self._get_element_text(person.find('.//tei:forename', namespaces=ns))
#             surname = self._get_element_text(person.find('.//tei:surname', namespaces=ns))
#             if forename or surname:
#                 full_name = f"{forename} {surname}".strip()
#                 if full_name:
#                     authors.append(full_name)

#         # Extract publication year
#         year = None
#         date_elem = root.find('.//tei:date[@type="published"]', namespaces=ns)
#         if date_elem is None:
#             date_elem = root.find('.//tei:date', namespaces=ns)
        
#         if date_elem is not None:
#             when_attr = date_elem.get('when')
#             if when_attr:
#                 try:
#                     year = int(when_attr[:4])
#                 except (ValueError, TypeError):
#                     pass

#         # Extract DOI
#         doi_elem = root.find('.//tei:idno[@type="DOI"]', namespaces=ns)
#         doi = self._get_element_text(doi_elem)

#         # Extract journal
#         journal_elem = root.find('.//tei:monogr//tei:title[@level="j"]', namespaces=ns)
#         journal = self._get_element_text(journal_elem)

#         # Extract abstract
#         # abstract_elem = root.find('.//tei:abstract', namespaces=ns)
#         # abstract = self._get_element_text(abstract_elem) if abstract_elem is not None else None
#         # abstract = None
#         # abstract_elem = root.find('.//tei:teiHeader/tei:profileDesc/tei:abstract', namespaces=ns)
#         # if abstract_elem is None:
#         #      abstract_elem = root.find('.//tei:teiHeader/tei:fileDesc/tei:div[@type="abstract"]', namespaces=ns)
#         # if abstract_elem is not None:
#         #     abstract = self._get_element_text(abstract_elem)
#         abstract = None
#         abstract_paths = [
#             './/tei:teiHeader//tei:abstract',
#             './/tei:profileDesc//tei:abstract', 
#             './/tei:abstract'
#         ]

#         for xpath in abstract_paths:
#             abstract_elem = root.find(xpath, namespaces=ns)
#             if abstract_elem is not None:
#                 # First try to get text directly
#                 abstract = self._get_element_text(abstract_elem)
                
#                 # If that fails, try extracting from paragraph children
#                 if not abstract or len(abstract.strip()) < 50:
#                     p_texts = []
#                     for p in abstract_elem.findall('.//tei:p', namespaces=ns):
#                         p_text = self._get_element_text(p)
#                         if p_text:
#                             p_texts.append(p_text)
#                     if p_texts:
#                         abstract = "\n\n".join(p_texts)
                
#                 # If we found a good abstract, break
#                 if abstract and len(abstract.strip()) > 50:
#                     break

#         # Reset to None if still too short
#         if abstract and len(abstract.strip()) < 50:
#             abstract = None




#         # Extract keywords
#         keywords = []
#         for keyword_elem in root.findall('.//tei:term', namespaces=ns):
#             keyword = self._get_element_text(keyword_elem)
#             if keyword:
#                 keywords.append(keyword)

#         return PaperMetadata(
#             title=title,
#             authors=authors,
#             publication_year=year,
#             doi=doi,
#             journal=journal,
#             abstract=abstract,
#             keywords=keywords
#         )

#     def _extract_structured_sections(self, root, ns: Dict[str, str]) -> List[StructuredSection]:
#         """Extract structured sections from TEI body"""
#         sections = []
        
#         # Find main body
#         body = root.find('.//tei:body', namespaces=ns)
#         if body is None:
#             return sections

#         # Extract main sections
#         for div in body.findall('.//tei:div', namespaces=ns):
#             section = self._parse_section(div, ns)
#             if section:
#                 sections.append(section)



#         # Extract back matter sections (funding, data availability, etc.)
#         back = root.find('.//tei:back', namespaces=ns)
#         if back is not None:
#             for div in back.findall('.//tei:div', namespaces=ns):
#                 section = self._parse_section(div, ns)
#                 if section:
#                     sections.append(section)
        
#         # Also check for any top-level divs outside body/back
#         for div in root.findall('.//tei:div', namespaces=ns):
#             # Skip if already processed (in body or back)
#             if (div.getparent() is not None and 
#                 div.getparent().tag.endswith('}body') or 
#                 div.getparent().tag.endswith('}back')):
#                 continue
            
#             section = self._parse_section(div, ns)
#             # if section and section.title.lower() in [
#             #     'data availability', 'data availability statement', 
#             #     'author contributions', 'funding', 'acknowledgments',
#             #     'acknowledgements', 'ethics statement', 'competing interests',
#             #     'conflict of interest', 'declarations'
#             # ]:
#             sections.append(section)

#         # Filter out duplicate sections by content - TODO: find better way to handle this
#         seen_content = set()
#         unique_sections = []
#         for section in sections:
#             content = getattr(section, 'content', None)
#             if not isinstance(content, str):
#                 # Skip sections without valid string content
#                 continue
#             normalized_content = ' '.join(content.split())
#             if normalized_content not in seen_content:
#                 seen_content.add(normalized_content)
#                 unique_sections.append(section)
#         return unique_sections

#     def _parse_section(self, div_elem, ns: Dict[str, str]) -> Optional[StructuredSection]:
#         """Parse a single section/div element"""
#         # Get section title
#         head_elem = div_elem.find('./tei:head', namespaces=ns)
#         title = self._get_element_text(head_elem) if head_elem is not None else "Untitled Section"
        
#         # Determine section type
#         section_type = self._classify_section_type(title)
        
#         # Extract content (paragraphs)
#         content_parts = []
#         for p in div_elem.findall('.//tei:p', namespaces=ns):
#             p_text = self._get_element_text(p)
#             if p_text:
#                 content_parts.append(p_text)
        
#         content = "\n\n".join(content_parts)
        
#         if not content.strip():
#             return None



#         # # Extract tables
#         tables = []
#         for table in div_elem.findall('.//tei:table', namespaces=ns):
#             table_head = self._get_element_text(table.find('./tei:head', namespaces=ns))
#             if table_head:
#                 tables.append({
#                     'caption': table_head,
#                     'content': 'Table content not extracted'  # Could be 
#                 })


#         # Extract reference citations in this section
#         references_cited = []
#         for ref in div_elem.findall('.//tei:ref[@type="bibr"]', namespaces=ns):
#             ref_target = ref.get('target')
#             if ref_target:
#                 references_cited.append(ref_target)

#         return StructuredSection(
#             section_type=section_type,
#             title=title,
#             content=content,
#             # subsections=subsections,
#             # figures=figures,
#             # tables=tables,
#             references_cited=references_cited
#         )




#     def _classify_section_type(self, title: str) -> str:
#         title_clean = title.lower().strip()
#         for bucket, patterns in COMPILED_PATTERNS.items():
#             if any(p.search(title_clean) for p in patterns):
#                 return bucket
#         return 'other'





#     def _extract_references(self, root, ns: Dict[str, str]) -> List[Reference]:
#         """Extract structured references from bibliography"""
#         references = []
        
#         # Find bibliography section
#         for div in root.findall('.//tei:div[@type="references"]', namespaces=ns):
#             for bibl in div.findall('.//tei:biblStruct', namespaces=ns):
#                 ref = self._parse_reference(bibl, ns)
#                 if ref:
#                     references.append(ref)
        
#         # Also check for listBibl
#         for list_bibl in root.findall('.//tei:listBibl', namespaces=ns):
#             for bibl in list_bibl.findall('.//tei:biblStruct', namespaces=ns):
#                 ref = self._parse_reference(bibl, ns)
#                 if ref:
#                     references.append(ref)

#         return references

#     def _parse_reference(self, bibl_elem, ns: Dict[str, str]) -> Optional[Reference]:
#         """Parse a single reference from biblStruct"""
#         # Extract title
#         title_elem = bibl_elem.find('.//tei:title[@level="a"]', namespaces=ns)
#         if title_elem is None:
#             title_elem = bibl_elem.find('.//tei:title', namespaces=ns)
#         title = self._get_element_text(title_elem)

#         # ISSUE WITH THIS ONE IS THAT IT COLLECTS ALSO AUTHORS FROM REFERECES (?)
#         # # Extract authors

#         authors = []
#         for person in bibl_elem.findall('.//tei:analytic/tei:author/tei:persName', namespaces=ns):
#             forename = self._get_element_text(person.find('.//tei:forename', namespaces=ns))
#             surname = self._get_element_text(person.find('.//tei:surname', namespaces=ns))
#             if forename or surname:
#                 full_name = f"{forename} {surname}".strip()
#                 if full_name:
#                     authors.append(full_name)


#         # Extract journal
#         journal_elem = bibl_elem.find('.//tei:title[@level="j"]', namespaces=ns)
#         journal = self._get_element_text(journal_elem)

#         # Extract year
#         year = None
#         date_elem = bibl_elem.find('.//tei:date', namespaces=ns)
#         if date_elem is not None:
#             when_attr = date_elem.get('when')
#             if when_attr:
#                 try:
#                     year = int(when_attr[:4])
#                 except (ValueError, TypeError):
#                     pass

#         # Extract DOI
#         doi_elem = bibl_elem.find('.//tei:idno[@type="DOI"]', namespaces=ns)
#         doi = self._get_element_text(doi_elem)

#         # Extract URL
#         url_elem = bibl_elem.find('.//tei:ptr', namespaces=ns)
#         url = url_elem.get('target') if url_elem is not None else None

#         # Create raw text representation
#         raw_parts = []
#         if authors:
#             raw_parts.append(", ".join(authors))
#         if title:
#             raw_parts.append(f'"{title}"')
#         if journal:
#             raw_parts.append(journal)
#         if year:
#             raw_parts.append(str(year))
        
#         raw_text = ". ".join(raw_parts) if raw_parts else "Reference not parsed"

#         # Check if this might be a data source
#         # is_data_source = self._is_potential_data_source(title, journal, raw_text)

#         return Reference(
#             raw_text=raw_text,
#             title=title,
#             authors=authors,
#             journal=journal,
#             year=year,
#             doi=doi,
#             url=url,
#             # is_data_source=is_data_source
#         )

#     # # TODO: IMPROVE OR REMOVE THIS FUNCTION
#     # # RATHER. this could/should be done based on dataset output
#     # def _is_potential_data_source(self, title: str, journal: str, raw_text: str) -> bool:
#     #     """Heuristically determine if reference might be a data source"""
#     #     if not title and not journal and not raw_text:
#     #         return False
            
#     #     text_to_check = f"{title or ''} {journal or ''} {raw_text}".lower()
        
#     #     data_keywords = [
#     #         'database', 'dataset', 'registry', 'surveillance', 'survey',
#     #         'census', 'repository', 'cohort', 'biobank', 'collection',
#     #         'who', 'cdc', 'nhanes', 'brfss', 'seer', 'eurostat'
#     #     ]
        
#     #     return any(keyword in text_to_check for keyword in data_keywords)

#     # def _get_element_text(self, elem) -> Optional[str]:
#     #     """Safely extract text from XML element"""
#     #     if elem is None:
#     #         return None
        
#     #     # Get all text content, including from child elements
#     #     text_parts = []
#     #     if elem.text:
#     #         text_parts.append(elem.text.strip())
        
#     #     for child in elem:
#     #         if child.text:
#     #             text_parts.append(child.text.strip())
#     #         if child.tail:
#     #             text_parts.append(child.tail.strip())
        
#     #     if elem.tail:
#     #         text_parts.append(elem.tail.strip())
        
#     #     full_text = " ".join(text_parts).strip()
#     #     return full_text if full_text else None


#     def _get_element_text(self, elem) -> Optional[str]:
#         """Safely extract text from XML element, preserving URLs and links"""
#         if elem is None:
#             return None
        
#         def extract_text_with_links(element):
#             """Recursively extract text while preserving links"""
#             parts = []
            
#             # Add element's direct text
#             if element.text:
#                 parts.append(element.text.strip())
            
#             # Process children
#             for child in element:
#                 if child.tag.endswith('}ptr'):  # TEI pointer element
#                     target = child.get('target')
#                     if target:
#                         parts.append(target)
#                 elif child.tag.endswith('}ref'):  # TEI reference element  
#                     target = child.get('target')
#                     child_text = extract_text_with_links(child)
#                     if target:
#                         parts.append(f"{child_text} ({target})" if child_text else target)
#                     elif child_text:
#                         parts.append(child_text)
#                 else:
#                     # Recursively process other elements
#                     child_text = extract_text_with_links(child)
#                     if child_text:
#                         parts.append(child_text)
                
#                 # Add tail text
#                 if child.tail:
#                     parts.append(child.tail.strip())
            
#             return ' '.join(parts).strip()
        
#         text = extract_text_with_links(elem)
        
#         if text:
#             # Clean up whitespace but preserve URLs
#             text = re.sub(r'\s+', ' ', text.strip())
#             return text if text else None
        
#         return None

#     def _create_fallback_data(self, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
#         """Create fallback data when GROBID fails"""
#         assert False, "GROBID failed"
#         metadata = PaperMetadata(
#             title=Path(pdf_path).stem,
#             authors=[],
#             publication_year=None,
#             doi=None
#         )
#         return metadata, [], []

