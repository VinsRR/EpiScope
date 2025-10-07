# EpiScope

EpiScope is a modular framework for evidence‑based question answering and
structured information extraction over scientific papers.  It brings
together retrieval‑augmented generation (RAG) and deep parsing to
support both high‑level exploration and detailed data mining.

## Features

* **Explorer mode** – Embed all papers in a project and answer
  questions with provenance using dense, sparse and late interaction
  embeddings stored in Qdrant.  Powered by local or cloud LLMs.
* **PrecisionMiner mode** – Parse individual PDFs via GROBID, extract
  tables, classify paper type, run LLM‑based extraction and build
  per‑paper FAISS indices.
* **Provenance** – All answers include citations: paper ID, snippet,
  section, index version and model/prompt identifiers.
* **Document loaders** – Flexible backends (Unstructured, GROBID or
  future options) convert local PDFs and text files into structured
  sections compatible with the indexers.
* **Pluggable providers** – Support for local (Ollama) and proprietary
  (OpenAI, Gemini) language models.  Configurable via environment.
* **REST API and CLI** – Access the system via FastAPI or a Typer
  command line interface.  A simple Streamlit UI is also available.
* **Project separation and snapshots** – Placeholder directories for
  organising data by project; future work will add snapshotting and
  versioning support.

## Quickstart

### Build and run via Docker

A `Dockerfile` and `docker-compose.yml` are provided for running the
API alongside Qdrant and GROBID.  To build and start the services:

```bash
docker compose up --build
```

The API will be available at http://localhost:8000 and GROBID at
http://localhost:8070 (if enabled).

### Install dependencies locally

EpiScope relies on Python 3.10+.  Install dependencies via pip:

```bash
python -m pip install -r requirements.txt
```

### Ingesting and querying via CLI

To ingest a PDF and ask a question from the command line:

```bash
python -m episcope.cli.episcope ingest --file-path /path/to/paper.pdf
python -m episcope.cli.episcope query "What is the basic reproduction number?" --top-k 3

### Loading documents for indexers

For workflows that require per‑paper indexing (e.g. PrecisionMiner),
you can convert a local file into structured sections and minimal
metadata using the document loader factory.  For example:

```python
from episcope.ingest import DocumentLoaderFactory

loader = DocumentLoaderFactory.get_loader("unstructured")
sections, metadata = loader.load("/path/to/paper.txt")
# Pass sections and metadata to PaperIndexer.index_paper()
```

The loader will automatically detect the file type and use the
appropriate backend (currently Unstructured for PDFs and plain text
parsers for other formats).  See `episcope/ingest/document_loader.py`
for details.

### GROBID extraction and persistence

The PrecisionMiner mode uses GROBID to obtain rich structured data
from PDFs.  The function
``episcope.ingest.grobid_pipeline.extract_paper`` will run GROBID
against a single PDF, parse the resulting TEI XML into sections,
metadata and references, and then persist each component into a
document store via the
:class:`episcope.storage.academic_db_manager.AcademicDBManager`.
Callers must supply a ``strategy_name`` to namespace extractions.
For example:

```python
from episcope.ingest.grobid_pipeline import extract_paper
from episcope.storage.academic_db_manager import AcademicDBManager

# Use an in‑memory database for testing
db = AcademicDBManager(use_in_memory=True)
extract_paper("/path/to/paper.pdf", strategy_name="Strategy_V1_GROBID_Standard", db=db)

# Retrieve the stored metadata
metadata = db.retrieve("paper", "metadata", "Strategy_V1_GROBID_Standard")
print(metadata["title"])
```

If a running MongoDB instance is available and ``pymongo`` is
installed, the manager will store documents in the
``AcademicCorpus.ExtractedDocuments`` collection instead of the
in‑memory fallback.  A unique index on
``(paper_id, data_type, strategy_name)`` prevents accidental
overwrites.  When MongoDB is unavailable the manager falls back
automatically and you can optionally persist the in‑memory store to a
JSON file via the ``backup_file`` parameter.
```

### Running the API

To start the API locally:

```bash
uvicorn episcope.src.api:app --reload --port 8000
```

You can then ingest and query via HTTP as documented in `docs/API.md`.

### Streamlit UI

A simple evidence‑first user interface is provided.  Run:

```bash
streamlit run episcope/ui/streamlit_app.py
```

Upload a PDF via the sidebar and ask questions in the main panel.

## Configuration

Global configuration is defined in `settings.py` and can be overridden
with environment variables.  See `docs/CONFIGS.md` for details.

## Roadmap

The current refactor lays the foundation for a clean, modular
codebase but leaves several areas for future development:

* Port the remaining RAG methods (ColPali and GraphRAG) into
  `retrieve/rag/`.
* Integrate the PDF download utilities into the ingestion pipeline to
  support DOI ingestion and PDF validation.
* Add snapshotting and versioning of project data under `projects/`.
* Expand the test suite to cover ingestion, parsing, indexing,
  retrieval and API endpoints.  Provide fixtures and mocks for
  external services.
* Optimise performance by adding asynchronous processing and caching.

Contributions are welcome!  Please file issues or pull requests on the
project repository.
