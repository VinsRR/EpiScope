import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .clients import fetch_doi_from_crossref
from .matching import ReferenceMatcher

# Optional dependencies
try:
    from gmft.auto import TableDetector, AutoTableFormatter
    from gmft.pdf_bindings import PyPDFium2Document
    HAS_GMFT = True
except ImportError:
    HAS_GMFT = False

try:
    from gmft_pymupdf import PyMuPDFDocument
    HAS_GMFT_PYMUPDF = True
except ImportError:
    HAS_GMFT_PYMUPDF = False

try:
    import img2table
    HAS_IMG2TABLE = True
except ImportError:
    HAS_IMG2TABLE = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables and build candidate strings for reference matching."""

    def __init__(self, pdf_path: Union[str, Path], output_root: Optional[Union[str, Path]] = None, debug: bool = False):
        self.pdf_path = str(pdf_path)
        self.paper_id = Path(pdf_path).stem
        self.base_dir = Path(pdf_path).parent
        self.output_root = Path(output_root) if output_root else self.base_dir
        self.tables_dir = self.output_root / f"{self.paper_id}_tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy().replace(r'^\s*$', pd.NA, regex=True).dropna(how='all').dropna(axis=1, how='all')
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()
        return df

    def build_candidates_from_tables(self, tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates_set = set()
        for t in tables:
            if t.get('ref_lines') is None:
                continue
            for line in t.get('ref_lines', []):
                candidates_set.add(line)
        
        return [
            {'candidate': c.strip(), 'page': None, 'table_index': None, 'row_index': None, 'source': None}
            for c in candidates_set if c and isinstance(c, str)
        ]

    def extract_gmft_tables(self) -> List[Dict]:
        if not HAS_GMFT:
            logger.info("gmft not installed, skipping GMFT extraction.")
            return []
        
        results = []
        doc = None
        try:
            doc = PyMuPDFDocument(self.pdf_path) if HAS_GMFT_PYMUPDF else PyPDFium2Document(self.pdf_path)
            detector = TableDetector()
            formatter = AutoTableFormatter()

            for page_number in range(len(doc)):
                page = doc[page_number]
                cropped_tables = detector.extract(page)
                for t_idx, cropped_table in enumerate(cropped_tables):
                    try:
                        df = formatter.extract(cropped_table).df()
                        if df is None or df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                            continue
                        df = self._clean_df(df)
                        if df.empty:
                            continue
                        
                        csv_path = self.tables_dir / f"{self.paper_id}_gmft_p{page_number + 1}_t{t_idx}.csv"
                        df.to_csv(str(csv_path), index=False)
                        results.append({
                            "page": page_number + 1,
                            "table_index": t_idx,
                            "data": df.fillna("").values.tolist(),
                            "csv_path": str(csv_path),
                            "source": "gmft"
                        })
                    except Exception as te:
                        logger.warning(f"gmft: failed to process table p={page_number + 1} t={t_idx}: {te}")
        except Exception as e:
            logger.warning(f"gmft extraction failed: {e}")
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
        logger.info(f"gmft extracted {len(results)} tables")
        return results

    def extract_img2table_tables(self) -> List[Dict]:
        if not HAS_IMG2TABLE:
            logger.info("img2table not installed, skipping.")
            return []

        import cv2
        import fitz
        import layoutparser as lp
        from img2table.ocr import TesseractOCR

        results = []
        model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet")
        ocr = TesseractOCR(n_threads=1, lang="eng", psm=11)
        pdf = fitz.open(self.pdf_path)

        for page_num in tqdm(range(len(pdf)), desc="Processing PDF pages with img2table"):
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=600)
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            layout = model.detect(img_rgb)
            table_regions = [b for b in layout if b.type.lower() == "table"]

            for region_idx, table_region in enumerate(table_regions):
                x1, y1, x2, y2 = map(int, table_region.coordinates)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                tables = self._extract_tables_from_crop(crop, ocr)
                for table_idx, df in enumerate(tables):
                    if df.empty:
                        continue
                    
                    csv_name = f"{self.paper_id}_p{page_num}_r{region_idx}_t{table_idx}.csv"
                    csv_path = self.tables_dir / csv_name
                    df.to_csv(str(csv_path), index=False)
                    
                    ref_lines = self._extract_references_with_ollama(df)
                    results.append({
                        "page": page_num, "region": region_idx, "table": table_idx,
                        "data": df.fillna("").values.tolist(), "csv_path": str(csv_path),
                        "ref_lines": ref_lines, "source": "layoutparser+img2table"
                    })
        pdf.close()
        logger.info(f"Extracted {len(results)} tables with img2table")
        return results

    def _extract_tables_from_crop(self, crop_img, ocr):
        from img2table.document import Image
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
            tmp_path = tmpf.name
        cv2.imwrite(tmp_path, crop_img)
        
        try:
            doc = Image(tmp_path, detect_rotation=False)
            extracted = doc.extract_tables(ocr=ocr, implicit_rows=True, borderless_tables=True, min_confidence=30)
            return [table.df for table in extracted if table.df is not None and not table.df.empty]
        except Exception as e:
            logger.debug(f"Table extraction from crop failed: {e}")
            return []
        finally:
            os.unlink(tmp_path)

    def _extract_references_with_ollama(self, df: pd.DataFrame):
        if not HAS_OLLAMA:
            logger.warning("ollama is not installed. Cannot extract references from tables.")
            return []
        
        client = ollama.Client()
        try:
            response = client.chat(
                model="qwen2.5vl:3b",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting references from OCR tables. Respond ONLY with the requested list in plain text, one item per line. If none, respond with <no references>."},
                    {"role": "user", "content": f"Here is a table:\n\n{df.to_csv(index=False)}\n\nExtract author surnames as short paper references. Output only the references, one per line. Return all unique surnames you can find."},
                ],
                options={"temperature": 0}
            )
            content = response['message']['content']
            if "<no references>" not in content.lower():
                return [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Ollama reference extraction failed: {e}")
        return []

    def run_extractors(self) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        tables.extend(self.extract_gmft_tables())
        tables.extend(self.extract_img2table_tables())
        return tables

    def run_with_matcher(self, references: Sequence[Union[Dict, Any]], matcher: Optional[ReferenceMatcher] = None,
                        extracted_tables: Optional[List[Dict[str, Any]]] = None,
                        crossref_email: str = "you@example.com") -> Dict[str, Any]:
        if matcher is None:
            matcher = ReferenceMatcher()

        tables = extracted_tables if extracted_tables is not None else self.run_extractors()
        candidates = self.build_candidates_from_tables(tables)

        try:
            matcher.prepare(list(references))
            raw_matches = matcher.match_candidates(candidates, list(references), top_n=1)
        except Exception as e:
            logger.exception('Matching failed in run_with_matcher')
            raw_matches = []

        ref_map = {self._safe_get_reference_field(r, "index", i): r for i, r in enumerate(references)}
        comparison_results = [self._process_match(c, m, ref_map, crossref_email) for c, m in zip(candidates, raw_matches)]

        output = {
            "paper_id": self.paper_id,
            "summary": {
                "total_candidates": len(candidates),
                "candidates_with_matches": sum(1 for r in comparison_results if r["has_match"]),
                "crossref_dois_found": sum(1 for r in comparison_results if r["crossref_doi"]),
            },
            "tables": tables,
            "comparison_results": comparison_results
        }

        self._save_results(output)
        return output

    def _process_match(self, cand_obj, matches, ref_map, email):
        candidate_text = cand_obj.get('candidate', '')
        result = {"candidate_text": candidate_text, "has_match": False, "match_score": 0.0}
        
        if matches:
            top_match = matches[0]
            matched_index = top_match.get("matched_index")
            if matched_index is not None:
                result.update({"has_match": True, "match_score": top_match.get("score", 0.0), "reference_index": matched_index})
                ref_obj = ref_map.get(matched_index)
                if ref_obj:
                    result.update(self._get_ref_fields(ref_obj))
                    self._enrich_with_crossref(result, candidate_text, email)
        return result

    def _get_ref_fields(self, ref_obj):
        return {
            "reference_title": self._safe_get_reference_field(ref_obj, "title"),
            "reference_authors": self._safe_get_reference_field(ref_obj, "authors"),
            "reference_journal": self._safe_get_reference_field(ref_obj, "journal"),
            "reference_year": self._safe_get_reference_field(ref_obj, "year"),
            "reference_doi": self._safe_get_reference_field(ref_obj, "doi"),
        }

    def _enrich_with_crossref(self, result, candidate_text, email):
        if result.get("reference_doi"):
            result["crossref_doi"] = result["reference_doi"]
            result["crossref_score"] = 1.0
            return

        try:
            cr_res = fetch_doi_from_crossref(
                title=result["reference_title"] or candidate_text,
                authors=result["reference_authors"],
                journal=result["reference_journal"],
                year=result["reference_year"],
                user_agent_email=email,
            )
            result.update({
                "crossref_doi": cr_res.get("doi"),
                "crossref_score": cr_res.get("score", 0.0),
                "crossref_error": cr_res.get("error"),
            })
            if cr_res.get("item"):
                item = self._format_crossref_item(cr_res["item"])
                result["crossref_title"] = item.get("title")
        except Exception as e:
            logger.warning(f"Crossref lookup failed: {e}")
            result["crossref_error"] = str(e)

    def _safe_get_reference_field(self, ref_obj: Any, field: str, default=None):
        if hasattr(ref_obj, field):
            return getattr(ref_obj, field, default)
        if isinstance(ref_obj, dict):
            return ref_obj.get(field, default)
        return default

    def _format_crossref_item(self, item: Dict) -> Dict:
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
        year_parts = item.get("issued", {}).get("date-parts", [[]])[0]
        return {
            "title": (item.get("title") or [""])[0],
            "authors": authors,
            "journal": (item.get("container-title") or [""])[0],
            "year": str(year_parts[0]) if year_parts else "",
            "doi": item.get("DOI", ""),
        }

    def _save_results(self, output: Dict):
        try:
            summary_path = self.tables_dir / f"{self.paper_id}_match_comparison.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, default=str)
            
            df = pd.DataFrame(output["comparison_results"])
            df.to_csv(self.tables_dir / f"{self.paper_id}_comparison_results.csv", index=False)

            dois = [r["crossref_doi"] for r in output["comparison_results"] if r.get("crossref_doi")]
            with open(self.tables_dir / f"{self.paper_id}_matched_dois.txt", 'w') as f:
                f.write("\n".join(dois))
            logger.info(f"Results saved to {self.tables_dir}")
        except Exception as e:
            logger.warning(f'Failed to save results: {e}')
