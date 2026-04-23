# EpiScope

EpiScope is a Python package for retrieval-backed analysis of scientific papers, with a focus on evidence-aware workflows over epidemiology and public-health literature.

Today, the codebase is centered around three main entry points:

- a FastAPI backend in `src/episcope/api.py`
- a Streamlit UI in `ui/streamlit_app.py`
- reusable Python workflows under `src/episcope/workflows`

## What Is In The Package Today

The current `src/episcope` package includes:

- `api.py`: FastAPI app with `/health`, `/explore`, `/classify`, and `/precision-miner`
- `clients.py`: provider clients for Gemini, OpenAI, OpenRouter, and Ollama
- `db/`: academic paper storage interfaces plus MongoDB and in-memory implementations
- `rag/`: retrieval, query transformation, fusion, reranking, ingestion, embeddings, and generation components
- `schemas/`: paper metadata, sections, references, provenance, and search result schemas
- `vectordb/`: vector DB abstractions and Qdrant / FAISS-backed helpers
- `workflows/classification/`: evidence-backed paper classification workflows
- `workflows/precision_miner/`: targeted extraction workflows over retrieved evidence
- `finetuning/`: trace capture, review, repository, and JSONL export utilities for supervised fine-tuning data
- `settings.py`: environment-driven runtime configuration

The main workflow families currently exposed by the package are:

- `PaperClassifier`
- `PrecisionMiner`

The currently supported classification configs are:

- `PaperTypeClassifierConfig`
- `DataAccessibilityClassifierConfig`
- `DataTypeClassifierConfig`
- `GeoClassifierConfig`

The currently supported precision-miner configs are:

- `FindDataSourcesConfig`
- `FindSupplementaryLinksConfig`
- `IdentifyKeyReferencesConfig`

## Repository Layout

This repository now uses a standard `src` layout:

```text
EpiScope/
├── src/
│   └── episcope/
│       ├── api.py
│       ├── clients.py
│       ├── settings.py
│       ├── db/
│       ├── finetuning/
│       ├── rag/
│       ├── schemas/
│       ├── vectordb/
│       └── workflows/
├── tests/
├── ui/
├── docs/
├── notebooks/
├── deploy/
├── tableref/
├── pyproject.toml
└── README.md
```

Notes:

- `tableref/` is a sibling packaged component in the repo, not part of the main `episcope` package.
- `tests/` contains the active smoke, unit, and integration suites.
- `docs/` and `notebooks/` contain supporting material, but the source of truth for behavior is the code in `src/episcope/`.

## Installation

EpiScope requires Python 3.10+.

For local development, the most reliable setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Why both files?

- `pyproject.toml` contains the packaging metadata for the installable `episcope` package
- `requirements.txt` is currently the more complete local runtime/development stack, including API, UI, retrieval, and provider integrations

If you only want to build the package from the repository root:

```bash
python -m pip install build
python -m build
```

## Configuration

Runtime configuration is read from environment variables, with `.env` loaded automatically if present.

Common settings:

- `EPISCOPE_STRATEGY_NAME`
- `MONGO_URI`
- `MONGO_DB_NAME`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `EPISCOPE_LLM_PROVIDER`
- `EPISCOPE_LLM_MODEL`
- `CROSS_ENCODER_MODEL`
- `OLLAMA_HOST`
- `EPISCOPE_API_HOST`
- `EPISCOPE_API_PORT`
- `EPISCOPE_API_BASE_URL`
- `EPISCOPE_LOG_LEVEL`

Provider-specific credentials:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENROUTER_API_KEY`

Current provider values supported by the API and UI:

- `gemini`
- `openai`
- `openrouter`
- `ollama`

The default runtime settings live in `src/episcope/settings.py`.

## Running The API

Start the FastAPI app from the repository root:

```bash
uvicorn episcope.api:app --reload --host 0.0.0.0 --port 8000
```

The current API exposes:

- `GET /health`
- `POST /explore`
- `POST /classify`
- `POST /precision-miner`

Example health check:

```bash
curl http://localhost:8000/health
```

Example explorer request:

```bash
curl -X POST http://localhost:8000/explore \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What data sources were used in this study?",
    "top_k": 5,
    "generate_answer": true
  }'
```

Example classification request:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "paper-123",
    "classifier_kind": "data_accessibility",
    "detailed": true
  }'
```

Practical runtime notes:

- `/explore` expects an existing Qdrant collection
- `/classify` and `/precision-miner` expect both Qdrant and a configured MongoDB backend
- the request-level `config` object can override provider, model, retrieval mode, reranking, and backend settings per call

## Running The UI

Start the Streamlit app from the repository root:

```bash
streamlit run ui/streamlit_app.py
```

The current UI includes three tabs:

- `Explorer`
- `Classification`
- `Precision Miner`

The UI talks to the FastAPI backend and exposes the same main runtime controls:

- backend URLs and collection names
- LLM provider and model
- retrieval mode
- optional evidence reranking
- workflow kind selection for classification and precision-miner tasks

## Running With Docker Compose

The root [docker-compose.yml](/Users/vins/Documents/Projects/EpiScope/docker-compose.yml) supports three practical modes.

### 1. Local Dev Mode

This is the simplest setup and the best default for normal local development.

It starts:

- `qdrant`
- `api`
- `ui`

Run:

```bash
docker compose up --build
```

Direct local URLs:

- Qdrant: `http://127.0.0.1:6333`
- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:8501`

Use this mode when:

- you are developing locally
- you do not need a reverse proxy
- you want the smallest Docker setup in this repo

### 2. Local Or Server Mode With Reverse Proxy

This adds Caddy in front of the API and UI.

It starts:

- `qdrant`
- `api`
- `ui`
- `proxy`

Run:

```bash
docker compose --profile deploy up -d --build
```

Default proxy URL:

- `http://127.0.0.1:8080`

Routing behavior:

- `/` goes to the Streamlit UI
- `/api/...` goes to the FastAPI backend
- `/health` goes to the FastAPI backend

Use this mode when:

- you want one stable entrypoint instead of separate API/UI ports
- you want to mimic a deployment layout locally
- you want to expose only the proxy on a server

For a server-facing Linux deployment, you can publish the proxy more broadly:

```bash
EPISCOPE_PROXY_BIND_ADDRESS=0.0.0.0 \
EPISCOPE_PROXY_PORT=80 \
docker compose --profile deploy up -d --build
```

### 3. Proxy Mode With Optional Cloudflare Tunnel

This keeps the same proxy-based layout, but also starts `cloudflared`.

Run:

```bash
docker compose --profile tunnel up -d
```

This profile implicitly includes the proxy stack and adds:

- `cloudflared`

Use this mode when:

- you want to share a local/private deployment temporarily
- you want a quick public URL without managing a domain

### What The Compose Stack Does Not Start

The Docker Compose setup still does not provide:

- MongoDB
- GROBID

That means:

- `/explore` can work once Qdrant contains indexed data
- `/classify` and `/precision-miner` still need an external Mongo instance via `MONGO_URI`
- GROBID-backed PDF parsing still requires a separate GROBID setup if you want that path specifically

### Quick Summary

- `docker compose up --build`
  Best for simple local development with direct ports
- `docker compose --profile deploy up -d --build`
  Best for proxy-based local testing or a server deployment pattern
- `docker compose --profile tunnel up -d`
  Best for temporary external sharing through Cloudflare

## Python Usage

The public Python imports that are currently covered by the smoke test include:

```python
from episcope.workflows import PaperClassifier, PrecisionMiner
from episcope.workflows.classification import (
    ClassificationDecision,
    DataAccessibilityClassifierConfig,
    DetailedClassificationResult,
)
from episcope.workflows.precision_miner import (
    DetailedExtractionResult,
    FindDataSourcesConfig,
)
from episcope.db import get_academic_db
```

The document loader entry point is:

```python
from episcope.rag.ingestion.document_loader import DocumentLoaderFactory

loader = DocumentLoaderFactory.get_loader("unstructured")
sections, metadata, references = loader.load("/path/to/paper.pdf")
```

The current workflow objects are designed to be composed from:

- a retriever
- a generator
- a strategy name
- an academic DB backend when metadata should be resolved by paper id

For example, classification and precision-miner workflows are typically constructed around:

- `episcope.rag.retrieval.Retriever`
- `episcope.rag.generation.llm_generator.LLMGenerator`
- `episcope.vectordb.qdrant.QdrantDB`
- `episcope.db.MongoAcademicDB` or `episcope.db.get_academic_db(...)`

## Finetuning Utilities

The `episcope.finetuning` package is now part of the repo and supports a trace-review workflow for classifier outputs.

Key pieces:

- `TrainingCaptureSink`: persist completed classification traces
- `TrainingRepository`: store and load raw/reviewed records
- `TraceReviewer`: review captured traces and approve them into buckets
- `SFTExporter`: export approved records into JSONL training data

This is currently covered by `tests/unit/finetuning/test_capture_review_export.py`.

## Testing

After installing the package, run:

```bash
pytest -q
```

If you want to run tests from a checkout without installing the package first:

```bash
PYTHONPATH=src pytest -q
```

Useful entry points:

- `tests/smoke/test_public_imports.py`
- `tests/TESTING_GUIDE.md`

## Current Status Notes

A few parts of the repository are in transition:

- the FastAPI app, Streamlit UI, workflows, and `src/episcope` package structure are the clearest current entry points
- the `src/episcope/episcope.py` Typer CLI module is still present, but it points at older internal paths and should be treated as legacy until it is refreshed
- some older docs and notebooks may still reflect pre-flattening or pre-refactor module names

When in doubt, prefer the package modules under `src/episcope/` and the tested imports in `tests/`.
