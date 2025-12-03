# Reproducible Paper Extraction Pipeline

This directory contains the reorganized and documented code for the paper extraction pipeline. The goal of this refactoring is to ensure the reproducibility and maintainability of the process.

## Directory Structure

The code is organized into the following directories, each responsible for a specific part of the pipeline:

- `database/`: Contains the database schema (`schema.sql`) and utility functions (`db_utils.py`) for interacting with the SQLite database.
- `pipeline/`: The core logic of the review expansion pipeline.
  - `run_pipeline.py`: Processes review papers to extract references.
  - `paper_downloader.py`: Downloads the papers identified in the review expansion.
- `ingestion/`: Scripts for ingesting the downloaded papers.
  - `ingest_papers.py`: Uses Grobid to process PDFs and store them in a database (e.g., MongoDB).
- `data_preparation/`: Scripts for preparing and cleaning up data.
  - `prepare_and_backfill.py`: Fills in missing metadata for references using the Crossref API.
- `supplementary_handling/`: Tools for processing supplementary materials.
  - `process_supplementary.py`: Extracts references from supplementary documents.
- `scripts/`: Orchestration scripts to run the full pipeline.
  - `run_full_pipeline.py`: The main entry point to run all steps of the pipeline in sequence.

## How to Run the Pipeline

### Prerequisites

1.  **Python Environment**: Ensure you have a Python environment with the necessary libraries installed. You can typically find these in a `requirements.txt` file (not provided in the original code, but would be a good addition). Key libraries include `SQLAlchemy`, `requests`, `beautifulsoup4`, `PyMuPDF`, `PyPaperBot`, and the `episcope` and `tableref` packages.
2.  **Grobid**: A running Grobid instance is required for the ingestion step. You can run Grobid using Docker.
3.  **MongoDB**: A running MongoDB instance is needed to store the ingested paper data.
4.  **Environment Variables**: Set the following environment variables, especially for downloading papers from Open Access sources:
    ```bash
    export UNPAYWALL_EMAIL='your.email@example.com'
    # export CORE_API_KEY='your_core_api_key' # Optional
    ```

### Running the Full Pipeline

The entire pipeline can be executed by running the main orchestration script:

```bash
python -m repro_paper_extraction.scripts.run_full_pipeline
```

This will execute the following steps in order:

1.  **Review Expansion**: Process the review papers to identify references to other studies and store these in the `bibliography.db` SQLite database.
2.  **Metadata Backfilling**: Query the Crossref API to fill in any missing metadata for the extracted references.
3.  **Paper Downloading**: Download the PDF files for the extracted references.
4.  **Paper Ingestion**: Process the downloaded PDFs with Grobid and save the structured data to the MongoDB database.

By following this structure and documentation, the pipeline should be much easier to understand, maintain, and reproduce.
