# -*- coding: utf-8 -*-
"""
Orchestration script for the full paper extraction and ingestion pipeline.

This script runs the entire pipeline in sequence, from processing reviews
to downloading papers, and finally ingesting them into the database.
It is intended to be the main entry point for running the pipeline.
"""
import logging
from pathlib import Path

# Import the functions from the different pipeline modules
from ..pipeline.run_pipeline import setup_database, setup_pipeline_and_provenance, process_review
from ..pipeline.paper_downloader import process_reviews
from ..ingestion.ingest_papers import ingest_papers
from ..data_preparation.prepare_and_backfill import backfill_crossref_for_unmatched

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths and settings
DB_PATH = "bibliography.db"
ACADEMIC_DB_PATH = "full_WP2_db.json" # The JSON file with academic data
PAPERS_OUTPUT_DIR = "expanded_papers"
PROCESSED_LOG_PATH = Path("processed_papers.log")

# --- Main Pipeline ---

def run_full_pipeline():
    """
    Executes the full paper extraction and ingestion pipeline.
    """
    # --- Step 1: Run the review expansion pipeline ---
    # This step processes the review papers and extracts the references.
    logger.info("--- Starting Step 1: Review Expansion Pipeline ---")
    
    # List of review paper IDs to process
    review_paper_ids_to_process = [
        "2019_Beer_r0_basic_",
        "2020_Lin_r0_basic_",
        "2021_Alene_incubation_period",
        "2024_Nash_r0",
        "crimean_congo_hemorrhagic_fever__crimean-congo_hemorrhagic_fever_orthonairovirus",
        "2022_Belhadi_cfr",
    ]

    engine, tables = setup_database(DB_PATH)
    pipeline, academic_db, provenance = setup_pipeline_and_provenance(ACADEMIC_DB_PATH)

    for paper_id in review_paper_ids_to_process:
        try:
            process_review(paper_id, academic_db, pipeline, engine, tables, provenance)
        except Exception as e:
            logger.error(f"Failed to process review {paper_id}: {e}", exc_info=True)
    
    logger.info("--- Finished Step 1: Review Expansion Pipeline ---")

    # --- Step 2: Backfill metadata for unmatched references ---
    # This step enriches the database with metadata from Crossref.
    logger.info("--- Starting Step 2: Backfill Crossref Metadata ---")
    backfill_crossref_for_unmatched(db_path=DB_PATH)
    logger.info("--- Finished Step 2: Backfill Crossref Metadata ---")

    # --- Step 3: Download the extracted papers ---
    # This step downloads the PDFs of the papers that were extracted.
    logger.info("--- Starting Step 3: Download Extracted Papers ---")
    process_reviews(
        review_ids=review_paper_ids_to_process,
        base_output_dir=PAPERS_OUTPUT_DIR,
        db_path=DB_PATH,
    )
    logger.info("--- Finished Step 3: Download Extracted Papers ---")

    # --- Step 4: Ingest the downloaded papers ---
    # This step processes the downloaded PDFs with Grobid and saves them.
    logger.info("--- Starting Step 4: Ingest Downloaded Papers ---")
    ingest_papers(
        papers_directory=Path(PAPERS_OUTPUT_DIR),
        processed_log_path=PROCESSED_LOG_PATH,
        grobid_url="http://127.0.0.1:8070", # Adjust if needed
        mongo_uri="mongodb://localhost:27017/", # Adjust if needed
        mongo_db_name="episcope_academic_db",
    )
    logger.info("--- Finished Step 4: Ingest Downloaded Papers ---")

    logger.info("--- Full pipeline finished successfully! ---")

if __name__ == "__main__":
    run_full_pipeline()
