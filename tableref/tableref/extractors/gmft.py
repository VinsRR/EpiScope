import logging
from typing import List, Dict, Any
import pandas as pd

from .base import BaseExtractor

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

logger = logging.getLogger(__name__)

class GmftExtractor(BaseExtractor):
    """Extracts tables using the GMFT library."""

    def __init__(self):
        if not HAS_GMFT:
            raise ImportError("gmft is not installed. Please install it with 'pip install gmft[full]'")

    def extract(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        results = []
        doc = None
        try:
            doc = PyMuPDFDocument(pdf_path) if HAS_GMFT_PYMUPDF else PyPDFium2Document(pdf_path)
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

                        csv_path = f"{output_dir}/gmft_p{page_number + 1}_t{t_idx}.csv"
                        df.to_csv(csv_path, index=False)

                        results.append({
                            "page": page_number + 1,
                            "table_index": t_idx,
                            "data": df.fillna("").values.tolist(),
                            "df": df,
                            "csv_path": csv_path,
                            "source": "gmft"
                        })
                    except Exception as te:
                        logger.warning(f"gmft: failed to process table p={page_number + 1} t={t_idx}: {te}")
        except Exception as e:
            logger.warning(f"gmft extraction failed for {pdf_path}: {e}")
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
        logger.info(f"gmft extracted {len(results)} tables from {pdf_path}")
        return results

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy().replace(r'^\s*$', pd.NA, regex=True).dropna(how='all').dropna(axis=1, how='all')
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()
        return df
