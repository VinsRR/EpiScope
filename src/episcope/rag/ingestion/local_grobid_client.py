"""GROBID client for EpiScope.

This module provides a client for interacting with a GROBID server to
extract structured data from PDF files. It encapsulates the logic for
making requests to the GROBID service and parsing the resulting TEI
XML into the data blueprint classes defined in EpiScope.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree

from episcope.schemas import (
    StructuredSection,
    PaperMetadata,
    Reference
)

# at module load time
SECTION_KEYWORDS = {
    'introduction': [r'\bintroduction\b', r'\bbackground\b', r'\boverview\b',
                     r'\babstract\b', r'\bsummary\b',],
    'methods': [
        r'\bmethod(?:s)?\b', r'\bmaterial(?:s)?\b', r'\bapproach\b',
        r'\bexperiment(?:al)?\b', r'\bprocedure\b', r'\bdesign\b', r'\bmethodology\b'
    ],
    'results': [r'\bresult(?:s)?\b', r'\bfinding(?:s)?\b', r'\bobservation(?:s)?\b'],
    'discussion': [
        r'\bdiscussion\b', r'\bconclusion(?:s)?\b', r'\blimitation(?:s)?\b',
        r'\bfuture work\b', r'\bimplication(?:s)?\b'
    ],
    'declarations': [
        r'\backnowledg(?:ement|ments)\b', r'\bfunding\b',
        r'\bethic(?:s|al)\b', r'\bcompeting interest\b', r'\bconflict of interest\b'
    ],
    'references': [r'\breference(?:s)?\b', r'\bbibliography\b', r'\bworks cited\b'],
    'appendices': [
        r'\bappendix\b', r'\bannex\b',
        r'\bsupplementary\b', r'\bsupplemental\b',

    ],
    'data': [
        r'\bdata\b', r'\bdataset\b', r'\bsurvey\b', r'\bcensus\b',
    ],
    'code': [
        r'\bcode\b', r'\bsoftware\b'
    ],
    'untitled': [r'\buntitled\b']
}
COMPILED_PATTERNS = {
    bucket: [re.compile(pat) for pat in pats]
    for bucket, pats in SECTION_KEYWORDS.items()
}

logger = logging.getLogger(__name__)

from grobid_client.grobid_client import GrobidClient as PyClient
class GrobidClient:
    """ GROBID client that fully utilizes structured output"""

    def __init__(self, grobid_server: str = "http://localhost:8070", timeout: int = 3600):
        self.grobid_server = grobid_server
        self.timeout = timeout
        try:
            self.client = PyClient(grobid_server, timeout=timeout)
            logger.info(f" GROBID client initialized with server: {grobid_server}")
        except ImportError:
            logger.error("grobid_client_python not installed. Install with: pip install grobid_client_python")
            raise

    def process_fulltext(self, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
        """Process full document with GROBID and extract all structured information"""
        try:
            # Process the full document
            print(f"Processing fulltext for {pdf_path} with GROBID...")
            result = self.client.process_pdf(
                service="processFulltextDocument",
                pdf_file=pdf_path,
                generateIDs=True,
                consolidate_header=True,
                consolidate_citations=True,
                include_raw_citations=True,
                include_raw_affiliations=True,
                tei_coordinates=True,
                segment_sentences=True
            )

            # Handle different return formats
            if isinstance(result, tuple) and len(result) >= 3:
                xml_content = result[2]
            elif isinstance(result, str):
                xml_content = result
            else:
                logger.warning(f"Unexpected result format from GROBID: {type(result)}")
                return self._create_fallback_data(pdf_path)

            if not xml_content or not xml_content.strip().startswith('<'):
                logger.warning(f"No valid XML returned for {pdf_path}")
                return self._create_fallback_data(pdf_path)

            return self._parse_fulltext_tei(xml_content, pdf_path)

        except Exception as e:
            logger.error(f"GROBID fulltext processing failed for {pdf_path}: {e}")
            return self._create_fallback_data(pdf_path)

    def _parse_fulltext_tei(self, xml_content: str, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
        """Parse full TEI XML to extract comprehensive structured information"""
        try:
            root = etree.fromstring(xml_content.encode())

            # Handle namespaces
            nsmap = root.nsmap
            tei_ns = nsmap.get(None) or 'http://www.tei-c.org/ns/1.0'
            ns = {'tei': tei_ns}

            # Extract metadata
            metadata = self._extract__metadata(root, ns, pdf_path)

            # Extract structured sections
            sections = self._extract_structured_sections(root, ns)

            # Extract references
            references = self._extract_references(root, ns)

            logger.info(f"Extracted {len(sections)} sections and {len(references)} references from {pdf_path}")
            return metadata, sections, references

        except Exception as e:
            logger.error(f"Failed to parse fulltext TEI XML for {pdf_path}: {e}")
            return self._create_fallback_data(pdf_path)

    def _extract__metadata(self, root, ns: Dict[str, str], pdf_path: str) -> PaperMetadata:
        """Extract comprehensive metadata from TEI header"""
        # Extract title
        title_elem = root.find('.//tei:title[@level="a"]', namespaces=ns)
        if title_elem is None:
            title_elem = root.find('.//tei:title', namespaces=ns)
        title = self._get_element_text(title_elem) or Path(pdf_path).stem

        # Extract authors
        authors = []
        for person in root.findall('.//tei:author//tei:persName', namespaces=ns):
            forename = self._get_element_text(person.find('.//tei:forename', namespaces=ns))
            surname = self._get_element_text(person.find('.//tei:surname', namespaces=ns))
            if forename or surname:
                full_name = f"{forename} {surname}".strip()
                if full_name:
                    authors.append(full_name)

        # Extract publication year
        year = None
        date_elem = root.find('.//tei:date[@type="published"]', namespaces=ns)
        if date_elem is None:
            date_elem = root.find('.//tei:date', namespaces=ns)

        if date_elem is not None:
            when_attr = date_elem.get('when')
            if when_attr:
                try:
                    year = int(when_attr[:4])
                except (ValueError, TypeError):
                    pass

        # Extract DOI
        doi_elem = root.find('.//tei:idno[@type="DOI"]', namespaces=ns)
        doi = self._get_element_text(doi_elem)

        # Extract journal
        journal_elem = root.find('.//tei:monogr//tei:title[@level="j"]', namespaces=ns)
        journal = self._get_element_text(journal_elem)

        # Extract abstract
        abstract = None
        abstract_paths = [
            './/tei:teiHeader//tei:abstract',
            './/tei:profileDesc//tei:abstract',
            './/tei:abstract'
        ]

        for xpath in abstract_paths:
            abstract_elem = root.find(xpath, namespaces=ns)
            if abstract_elem is not None:
                # First try to get text directly
                abstract = self._get_element_text(abstract_elem)

                # If that fails, try extracting from paragraph children
                if not abstract or len(abstract.strip()) < 50:
                    p_texts = []
                    for p in abstract_elem.findall('.//tei:p', namespaces=ns):
                        p_text = self._get_element_text(p)
                        if p_text:
                            p_texts.append(p_text)
                    if p_texts:
                        abstract = "\n\n".join(p_texts)

                # If we found a good abstract, break
                if abstract and len(abstract.strip()) > 50:
                    break

        # Reset to None if still too short
        if abstract and len(abstract.strip()) < 50:
            abstract = None

        # Extract keywords
        keywords = []
        for keyword_elem in root.findall('.//tei:term', namespaces=ns):
            keyword = self._get_element_text(keyword_elem)
            if keyword:
                keywords.append(keyword)

        return PaperMetadata(
            title=title,
            authors=authors,
            publication_year=year,
            doi=doi,
            journal=journal,
            abstract=abstract,
            keywords=keywords
        )

    def _extract_structured_sections(self, root, ns: Dict[str, str]) -> List[StructuredSection]:
        """Extract structured sections from TEI body"""
        sections = []

        # Find main body
        body = root.find('.//tei:body', namespaces=ns)
        if body is None:
            return sections

        # Extract main sections
        for div in body.findall('.//tei:div', namespaces=ns):
            section = self._parse_section(div, ns)
            if section:
                sections.append(section)

        # Extract back matter sections (funding, data availability, etc.)
        back = root.find('.//tei:back', namespaces=ns)
        if back is not None:
            for div in back.findall('.//tei:div', namespaces=ns):
                section = self._parse_section(div, ns)
                if section:
                    sections.append(section)

        # Also check for any top-level divs outside body/back
        for div in root.findall('.//tei:div', namespaces=ns):
            # Skip if already processed (in body or back)
            if (
                (div.getparent() is not None and
                 (div.getparent().tag.endswith('}body') or
                  div.getparent().tag.endswith('}back')))
            ):
                continue

            section = self._parse_section(div, ns)
            if section:
                sections.append(section)

        # Filter out duplicate sections by content
        seen_content = set()
        unique_sections = []
        for section in sections:
            content = getattr(section, 'content', None)
            if not isinstance(content, str):
                continue
            normalized_content = ' '.join(content.split())
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_sections.append(section)
        return unique_sections

    def _parse_section(self, div_elem, ns: Dict[str, str]) -> Optional[StructuredSection]:
        """Parse a single section/div element"""
        head_elem = div_elem.find('./tei:head', namespaces=ns)
        title = self._get_element_text(head_elem) if head_elem is not None else "Untitled Section"

        section_type = self._classify_section_type(title)

        content_parts = []
        for p in div_elem.findall('.//tei:p', namespaces=ns):
            p_text = self._get_element_text(p)
            if p_text:
                content_parts.append(p_text)

        content = "\n\n".join(content_parts)

        if not content.strip():
            return None

        references_cited = []
        for ref in div_elem.findall('.//tei:ref[@type="bibr"]', namespaces=ns):
            ref_target = ref.get('target')
            if ref_target:
                references_cited.append(ref_target)

        return StructuredSection(
            section_type=section_type,
            title=title,
            content=content,
            references_cited=references_cited
        )

    def _classify_section_type(self, title: str) -> str:
        title_clean = title.lower().strip()
        for bucket, patterns in COMPILED_PATTERNS.items():
            if any(p.search(title_clean) for p in patterns):
                return bucket
        return 'other'

    def _extract_references(self, root, ns: Dict[str, str]) -> List[Reference]:
        """Extract structured references from bibliography"""
        references = []

        for div in root.findall('.//tei:div[@type="references"]', namespaces=ns):
            for bibl in div.findall('.//tei:biblStruct', namespaces=ns):
                ref = self._parse_reference(bibl, ns)
                if ref:
                    references.append(ref)

        for list_bibl in root.findall('.//tei:listBibl', namespaces=ns):
            for bibl in list_bibl.findall('.//tei:biblStruct', namespaces=ns):
                ref = self._parse_reference(bibl, ns)
                if ref:
                    references.append(ref)

        return references

    def _parse_reference(self, bibl_elem, ns: Dict[str, str]) -> Optional[Reference]:
        """Parse a single reference from biblStruct"""
        title_elem = bibl_elem.find('.//tei:title[@level="a"]', namespaces=ns)
        if title_elem is None:
            title_elem = bibl_elem.find('.//tei:title', namespaces=ns)
        title = self._get_element_text(title_elem)

        authors = []
        for person in bibl_elem.findall('.//tei:analytic/tei:author/tei:persName', namespaces=ns):
            forename = self._get_element_text(person.find('.//tei:forename', namespaces=ns))
            surname = self._get_element_text(person.find('.//tei:surname', namespaces=ns))
            if forename or surname:
                full_name = f"{forename} {surname}".strip()
                if full_name:
                    authors.append(full_name)

        journal_elem = bibl_elem.find('.//tei:title[@level="j"]', namespaces=ns)
        journal = self._get_element_text(journal_elem)

        year = None
        date_elem = bibl_elem.find('.//tei:date', namespaces=ns)
        if date_elem is not None:
            when_attr = date_elem.get('when')
            if when_attr:
                try:
                    year = int(when_attr[:4])
                except (ValueError, TypeError):
                    pass

        doi_elem = bibl_elem.find('.//tei:idno[@type="DOI"]', namespaces=ns)
        doi = self._get_element_text(doi_elem)

        url_elem = bibl_elem.find('.//tei:ptr', namespaces=ns)
        url = url_elem.get('target') if url_elem is not None else None

        raw_parts = []
        if authors:
            raw_parts.append(", ".join(authors))
        if title:
            raw_parts.append(f'"{title}"')
        if journal:
            raw_parts.append(journal)
        if year:
            raw_parts.append(str(year))

        raw_text = ". ".join(raw_parts) if raw_parts else "Reference not parsed"

        return Reference(
            raw_text=raw_text,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            doi=doi,
            url=url,
        )

    def _get_element_text(self, elem) -> Optional[str]:
        """Safely extract text from XML element, preserving URLs and links"""
        if elem is None:
            return None

        def extract_text_with_links(element):
            parts = []
            if element.text:
                parts.append(element.text.strip())
            for child in element:
                if child.tag.endswith('}ptr'):
                    target = child.get('target')
                    if target:
                        parts.append(target)
                elif child.tag.endswith('}ref'):
                    target = child.get('target')
                    child_text = extract_text_with_links(child)
                    if target:
                        parts.append(f"{child_text} ({target})") if child_text else target
                    elif child_text:
                        parts.append(child_text)
                else:
                    child_text = extract_text_with_links(child)
                    if child_text:
                        parts.append(child_text)
                if child.tail:
                    parts.append(child.tail.strip())
            return ' '.join(parts).strip()

        text = extract_text_with_links(elem)

        if text:
            text = re.sub(r'\s+', ' ', text.strip())
            return text if text else None

        return None

    def _create_fallback_data(self, pdf_path: str) -> Tuple[PaperMetadata, List[StructuredSection], List[Reference]]:
        """Create fallback data when GROBID fails"""
        logger.warning(f"Creating fallback data for {pdf_path}")
        metadata = PaperMetadata(
            title=Path(pdf_path).stem,
            authors=[],
            publication_year=None,
            doi=None
        )
        return metadata, [], []
