import logging
import os
from typing import List, Dict, Any

import cv2
import fitz  # PyMuPDF
import layoutparser as lp
import numpy as np
import pandas as pd
from tqdm import tqdm
from img2table.ocr import TesseractOCR
from img2table.document import Image as Img2TableImage

from .base import BaseExtractor

logger = logging.getLogger(__name__)

class Img2TableExtractor(BaseExtractor):
    """Extracts tables using a combination of layoutparser and img2table."""

    def __init__(self):
        self.model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet")
        self.ocr = TesseractOCR(n_threads=1, lang="eng", psm=11)

    def extract(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        results = []
        pdf = fitz.open(pdf_path)

        for page_num in tqdm(range(len(pdf)), desc=f"Processing {os.path.basename(pdf_path)} with img2table"):
            page = pdf[page_num]
            img = self._page_to_image(page)
            if img is None:
                continue

            layout = self.model.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            table_regions = [b for b in layout if b.type.lower() == "table"]

            for region_idx, table_region in enumerate(table_regions):
                x1, y1, x2, y2 = map(int, table_region.coordinates)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                tables = self._extract_tables_from_crop(crop)
                for table_idx, df in enumerate(tables):
                    if df.empty:
                        continue

                    csv_name = f"p{page_num}_r{region_idx}_t{table_idx}.csv"
                    csv_path = os.path.join(output_dir, csv_name)
                    df.to_csv(csv_path, index=False)

                    results.append({
                        "page": page_num, "region": region_idx, "table": table_idx,
                        "data": df.fillna("").values.tolist(),
                        "df": df,
                        "csv_path": csv_path,
                        "source": "layoutparser+img2table"
                    })
        pdf.close()
        logger.info(f"Extracted {len(results)} tables from {pdf_path} with img2table")
        return results

    def _page_to_image(self, page: fitz.Page) -> np.ndarray:
        pix = page.get_pixmap(dpi=300) # Use a reasonable DPI (Dots Per Inch)
        return cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    def _extract_tables_from_crop(self, crop_img: np.ndarray) -> List[pd.DataFrame]:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
            tmp_path = tmpf.name
        cv2.imwrite(tmp_path, crop_img)

        try:
            doc = Img2TableImage(tmp_path, detect_rotation=False)
            extracted = doc.extract_tables(
                ocr=self.ocr, implicit_rows=True, borderless_tables=True, min_confidence=30
            )
            return [table.df for table in extracted if table.df is not None and not table.df.empty]
        except Exception as e:
            logger.debug(f"Table extraction from crop failed: {e}")
            return []
        finally:
            os.unlink(tmp_path)
