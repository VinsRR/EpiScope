# -*- coding: utf-8 -*-
"""
Ingest downloaded papers into a database using Grobid.

This script scans a directory for PDF files, processes them with a Grobid
service to extract structured academic information (metadata, sections, references),
and then stores the results in a database.

It maintains a record of processed files to allow for resumable ingestion.
"""
import logging
from pathlib import Path
from datetime import datetime

# Note: Assumes 'episcope' is installed and available in the environment.
from episcope.rag.ingestion.document_loader import GrobidDocumentLoader
from episcope.db.mongo_academic_db import MongoAcademicDB

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# URL of your running Grobid instance
GROBID_URL = "http://127.0.0.1:8070" # Default, change if your Grobid is elsewhere

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017/" # Default, change to your MongoDB URI
MONGO_DB_NAME = "episcope_academic_db"

# Path to the directory containing downloaded papers
PAPERS_DIR = Path("expanded_papers")

# File to keep track of processed PDFs
PROCESSED_FILES_LOG = Path("processed_papers.log")

def ingest_papers(
    papers_directory: Path,
    processed_log_path: Path,
    grobid_url: str,
    mongo_uri: str,
    mongo_db_name: str,
    strategy_name: str = "grobid"
):
    """
    Processes PDF files in a directory with Grobid and saves them to a database.

    Args:
        papers_directory: The directory to scan for PDF files.
        processed_log_path: Path to a log file for tracking processed files.
        grobid_url: The URL of the Grobid service.
        mongo_uri: The connection URI for the MongoDB database.
        mongo_db_name: The name of the database to use.
        strategy_name: The name of the extraction strategy (defaults to "grobid").
    """
    # Initialize database connection
    try:
        db = MongoAcademicDB(uri=mongo_uri, db_name=mongo_db_name)
        # Test connection
        db.client.server_info()
        logger.info(f"Connected to MongoDB at {mongo_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return

    # Initialize Grobid loader
    loader = GrobidDocumentLoader(grobid_url)
    if not loader.is_alive():
        logger.error(f"Grobid service is not responding at {grobid_url}")
        return

    # Load the set of already processed files
    processed_log_path.touch(exist_ok=True)
    processed = set(
        line.strip()
        for line in processed_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    logger.info(f"Found {len(processed)} already processed papers in the log.")

    # Find all PDF files in the directory
    pdf_files = list(papers_directory.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} total PDF files to consider.")

    for pdf_path in pdf_files:
        pdf_path_str = str(pdf_path.resolve())
        if pdf_path_str in processed:
            logger.debug(f"Skipping already processed paper: {pdf_path_str}")
            continue

        logger.info(f"Processing: {pdf_path.name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            # The extract_paper method handles the interaction with Grobid and the DB
            loader.extract_paper(
                file_path=pdf_path,
                strategy_name=strategy_name,
                db=db
            )
            # If successful, log the file path
            with processed_log_path.open("a", encoding="utf-8") as f:
                f.write(pdf_path_str + "\n")
            processed.add(pdf_path_str)
            logger.info(f"Successfully processed and ingested {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error processing {pdf_path_str}: {e}", exc_info=True)

    logger.info("Ingestion process finished.")

if __name__ == "__main__":
    # This allows the script to be run directly
    ingest_papers(
        papers_directory=PAPERS_DIR,
        processed_log_path=PROCESSED_FILES_LOG,
        grobid_url=GROBID_URL,
        mongo_uri=MONGO_URI,
        mongo_db_name=MONGO_DB_NAME,
    )
