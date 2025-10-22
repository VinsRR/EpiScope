import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .candidates import OllamaCandidateGenerator
from .clients import fetch_doi_from_crossref
from .config import CrossrefConfig, MatcherConfig, OllamaConfig, SimpleSurnameMatcherConfig
from .extractors.base import BaseExtractor
from .matching import ReferenceMatcher

logger = logging.getLogger(__name__)

class TableRefPipeline:
    """
    Orchestrates the end-to-end process of table extraction, candidate generation,
    reference matching, and data enrichment.
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        candidate_generator: OllamaCandidateGenerator,
        matcher: ReferenceMatcher,
        crossref_config: Optional[CrossrefConfig] = None,
        output_root: Union[str, Path] = "output",
    ):
        self.extractors = extractors
        self.candidate_generator = candidate_generator
        self.matcher = matcher
        self.crossref_config = crossref_config or CrossrefConfig()
        self.output_root = Path(output_root)

    def run(self, pdf_path: Union[str, Path], references: Sequence[Union[Dict, Any]]) -> Dict[str, Any]:
        """
        Executes the full pipeline for a single PDF.

        Args:
            pdf_path: Path to the PDF file to process.
            references: A list of canonical reference objects to match against.

        Returns:
            A dictionary containing the final, structured results.
        """
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem
        paper_output_dir = self.output_root / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract tables using all configured extractors
        extracted_tables = []
        for extractor in self.extractors:
            extracted_tables.extend(extractor.extract(str(pdf_path), str(paper_output_dir)))

        # 2. Generate candidates from extracted tables
        for table in extracted_tables:
            df = table.get("df")
            if df is not None and not df.empty:
                table["ref_lines"] = self.candidate_generator.generate(df)
        
        candidates = self._build_candidates_from_tables(extracted_tables)

        # 3. Match candidates against references
        self.matcher.prepare(list(references))
        raw_matches = self.matcher.match_candidates(candidates, list(references), top_n=1)

        # 4. Process and enrich matches
        ref_map = {self._safe_get_reference_field(r, "index", i): r for i, r in enumerate(references)}
        comparison_results = [
            self._process_match(c, m, ref_map) for c, m in zip(candidates, raw_matches)
        ]

        # 5. Format and save final output
        output = self._format_output(paper_id, extracted_tables, candidates, comparison_results)
        self._save_results(output, paper_output_dir)

        return output

    def _build_candidates_from_tables(self, tables: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates_set = set()
        for t in tables:
            for line in t.get('ref_lines', []):
                candidates_set.add(line)
        return [{'candidate': c.strip()} for c in candidates_set if c and isinstance(c, str)]

    def _process_match(self, cand_obj, matches, ref_map):
        result = {"candidate_text": cand_obj.get('candidate', ''), "has_match": False, "match_score": 0.0}
        if matches:
            top_match = matches[0]
            matched_index = top_match.get("matched_index")
            if matched_index is not None:
                result.update({"has_match": True, "match_score": top_match.get("score", 0.0), "reference_index": matched_index})
                ref_obj = ref_map.get(matched_index)
                if ref_obj:
                    result.update(self._get_ref_fields(ref_obj))
                    self._enrich_with_crossref(result)
        return result

    def _get_ref_fields(self, ref_obj):
        return {
            "reference_title": self._safe_get_reference_field(ref_obj, "title"),
            "reference_authors": self._safe_get_reference_field(ref_obj, "authors"),
            "reference_journal": self._safe_get_reference_field(ref_obj, "journal"),
            "reference_year": self._safe_get_reference_field(ref_obj, "year"),
            "reference_doi": self._safe_get_reference_field(ref_obj, "doi"),
        }

    def _enrich_with_crossref(self, result):
        if result.get("reference_doi"):
            result["crossref_doi"] = result["reference_doi"]
            return
        try:
            cr_res = fetch_doi_from_crossref(
                title=result["reference_title"] or result["candidate_text"],
                config=self.crossref_config,
                authors=result["reference_authors"],
                journal=result["reference_journal"],
                year=result["reference_year"],
            )
            result.update({"crossref_doi": cr_res.get("doi"), "crossref_error": cr_res.get("error")})
        except Exception as e:
            logger.warning(f"Crossref lookup failed: {e}")
            result["crossref_error"] = str(e)

    def _safe_get_reference_field(self, ref_obj: Any, field: str, default=None):
        if hasattr(ref_obj, field):
            return getattr(ref_obj, field, default)
        if isinstance(ref_obj, dict):
            return ref_obj.get(field, default)
        return default

    def _format_output(self, paper_id, tables, candidates, comparison_results):
        # Remove DataFrame objects before serialization
        for table in tables:
            table.pop("df", None)
            
        return {
            "paper_id": paper_id,
            "summary": {
                "total_candidates": len(candidates),
                "candidates_with_matches": sum(1 for r in comparison_results if r["has_match"]),
                "crossref_dois_found": sum(1 for r in comparison_results if r.get("crossref_doi")),
            },
            "tables": tables,
            "comparison_results": comparison_results,
        }

    def _save_results(self, output: Dict, output_dir: Path):
        try:
            summary_path = output_dir / "match_comparison.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, default=str)
            
            df = pd.DataFrame(output["comparison_results"])
            df.to_csv(output_dir / "comparison_results.csv", index=False)

            dois = [r["crossref_doi"] for r in output["comparison_results"] if r.get("crossref_doi")]
            with open(output_dir / "matched_dois.txt", 'w') as f:
                f.write("\n".join(dois))
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.warning(f'Failed to save results: {e}')