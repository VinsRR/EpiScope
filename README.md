# EpiScope

EpiScope is a Python package for retrieval-backed analysis of scientific papers, with a focus on evidence-aware workflows over epidemiology and public-health literature.

Today, the codebase is centered around three main entry points:

- a FastAPI backend in `src/episcope/api.py`
- a Streamlit UI in `src/episcope/ui/streamlit_app.py`
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

These workflow *kinds* are registered in one place —
[`src/episcope/workflows/registry.py`](src/episcope/workflows/registry.py). The
CLI choices, the API request schema, and the Streamlit dropdowns all derive from
that registry, so adding a new classifier or miner is a single entry there (plus
its config and output schema), not edits spread across the CLI, API, runtime,
and UI. You can also add tasks **without writing Python** — see
[Defining Your Own Tasks](#defining-your-own-tasks).

## Defining Your Own Tasks

You can add a classifier or precision-miner by describing it as a JSON *task
spec* — no Python required. A classifier task lists its labels (each with a
`code`, `name`, `definition`, and optional example sentences that seed
retrieval); the model is told the valid codes (unknown codes are ignored when
parsing). A miner task lists retrieval prompts; its extraction schema is fixed. Prompts default to sensible
templates and can be overridden with `system_prompt` / `user_prompt_template`.

Example classifier task (`study_design.json`):

```json
{
  "key": "study_design",
  "kind": "classifier",
  "label": "Study design",
  "description": "Primary epidemiological study design.",
  "multi_label": false,
  "default_label": "unclear",
  "labels": [
    {"code": "cohort", "name": "Cohort", "definition": "Follows groups over time.",
     "examples": ["We followed a cohort of exposed individuals over 12 months."]},
    {"code": "case_control", "name": "Case-control", "definition": "Compares cases to controls.",
     "examples": ["Cases were matched to controls by age and sex."]},
    {"code": "cross_sectional", "name": "Cross-sectional", "definition": "A snapshot at one time point.",
     "examples": ["A cross-sectional survey was conducted in May 2020."]},
    {"code": "unclear", "name": "Unclear", "definition": "Not enough information."}
  ]
}
```

Example miner task (`find_funding.json`):

```json
{
  "key": "find_funding",
  "kind": "miner",
  "label": "Find funding sources",
  "description": "Funding bodies and grant numbers.",
  "section_filters": ["Acknowledgements", "Funding"],
  "retrieval_templates": [
    "Who funded this study?",
    "What grant or award numbers are reported?"
  ]
}
```

Load and run a task in any of these ways:

- **Workspace** (auto-discovered): drop the file in `<workspace>/tasks/` and run
  it by key.

  ```bash
  episcope classify --file paper.pdf --workspace my-review --classifier-kind study_design
  ```

- **Portable / inline** (works from any entrypoint, no setup): pass the file
  directly, or send the spec inline to the API.

  ```bash
  episcope classify --file paper.pdf --task-file study_design.json
  ```

  ```bash
  curl -X POST http://localhost:8000/classify \
    -H "Content-Type: application/json" \
    -d '{"paper_id": "paper-123", "task": { /* task spec */ }}'
  ```

- **Startup directory**: point `EPISCOPE_TASKS_DIR` at a folder of task JSON
  files; the CLI and API load them at startup, and they appear in `GET /health`
  and the Streamlit dropdowns.

List everything available (built-in and declarative):

```bash
episcope tasks
```

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
│       ├── rag/
│       ├── schemas/
│       ├── vectordb/
│       └── workflows/
├── tests/
├── ui/
├── docs/
├── notebooks/
├── deploy/
├── pyproject.toml
└── README.md
```

Notes:

- `tests/` contains the active smoke, unit, and integration suites.
- `docs/` and `notebooks/` contain supporting material, but the source of truth for behavior is the code in `src/episcope/`.

## Installation

EpiScope requires Python 3.10+.

EpiScope is not published on PyPI yet. Install it directly from GitHub:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "epi-scope @ git+https://github.com/VinsRR/EpiScope.git@main"
```

This installs the distribution named `epi-scope`; the Python import package is
still named `episcope`:

```bash
python -c "import episcope; print(episcope.__name__)"
```

For a reproducible environment, prefer pinning a tag or commit instead of the
moving `main` branch:

```bash
python -m pip install "epi-scope @ git+https://github.com/VinsRR/EpiScope.git@<tag-or-commit>"
```

For local development from a cloned checkout, use an editable install:

```bash
git clone https://github.com/VinsRR/EpiScope.git
cd EpiScope
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras can be installed from either GitHub or a local checkout:

```bash
python -m pip install "epi-scope[ui] @ git+https://github.com/VinsRR/EpiScope.git@main"
python -m pip install "epi-scope[server] @ git+https://github.com/VinsRR/EpiScope.git@main"
python -m pip install "epi-scope[dev] @ git+https://github.com/VinsRR/EpiScope.git@main"
```

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
- `EPISCOPE_EMBED_PROVIDER` (embedding provider: `auto` | `huggingface` | `openai` | `gemini` | `ollama`)
- `CROSS_ENCODER_MODEL`
- `OLLAMA_HOST`
- `EPISCOPE_API_HOST`
- `EPISCOPE_API_PORT`
- `EPISCOPE_API_BASE_URL`
- `EPISCOPE_LOG_LEVEL`
- `EPISCOPE_OUTPUT_FORMAT` (CLI default output format: `human` or `json`)

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

## Start Here For Epidemiologists

If you want to try EpiScope on a paper without setting up databases, use the CLI path first. It builds a temporary local index for the file or folder you provide, so you do not need MongoDB or Qdrant.

First, check that your environment is ready (Python version, LLM key, and optional services):

```bash
episcope doctor
```

It prints a checklist and exits non-zero if a required check fails. Use
`episcope doctor --no-probe` to skip the network probes. Every command prints a
human-readable summary by default; pass `--format json` (or set
`EPISCOPE_OUTPUT_FORMAT=json`) for machine-readable output. Run
`episcope --version` to print the installed version.

Inspect a document:

```bash
episcope inspect /path/to/paper.pdf
```

Ask a question over one paper or a folder of papers:

```bash
episcope ask "What data sources were used in this study?" --path /path/to/paper.pdf
```

Run retrieval without generating an answer:

```bash
episcope explore "What data sources were used?" --path /path/to/paper.pdf
```

Classify one paper:

```bash
episcope classify --file /path/to/paper.pdf --classifier-kind data_accessibility
```

Extract likely data sources:

```bash
episcope precision-miner --file /path/to/paper.pdf --miner-kind find_data_sources
```

Local notes:

- local indexing and retrieval default to `sentence-transformers/all-MiniLM-L6-v2`, which does not require a Gemini/OpenAI key
- answer generation and classification still require an LLM provider; use `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `--llm-provider ollama --llm-model <local-model>`
- use the API/UI/Docker path once you have a shared indexed corpus and want multiple users to work against the same backend

For a longer command-by-command walkthrough, see the
[wiki quickstart](https://github.com/VinsRR/EpiScope/wiki/Local-CLI-Quickstart).

## Workspaces

For repeated local work, create an EpiScope workspace. A workspace is a normal
folder that contains the project config, local metadata store, vector index,
outputs, and logs.

```bash
episcope init my-review
episcope index ./papers --workspace my-review
episcope papers --workspace my-review
episcope ask "Which papers mention GenBank?" --workspace my-review
```

This creates:

```text
my-review/
  episcope.toml
  papers/
  metadata.json
  index/
  outputs/
  logs/
```

If you run commands from inside the workspace directory, EpiScope discovers the
nearest `episcope.toml` automatically:

```bash
cd my-review
episcope papers
episcope ask "Which studies use surveillance data?"
```

The workspace file stores local paths and defaults. Secrets such as
`GEMINI_API_KEY`, `OPENAI_API_KEY`, and `MONGO_URI` should stay in your shell
environment or local `.env`, not in `episcope.toml`.

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
episcope-ui
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

The root [docker-compose.yml](docker-compose.yml) supports three practical modes.

### 1. Local Dev Mode

This is the simplest setup and the best default for normal local development.

It starts:

- `qdrant`
- `grobid`
- `api`
- `ui`

Run:

```bash
docker compose up --build
```

Direct local URLs:

- Qdrant: `http://127.0.0.1:6333`
- GROBID API: `http://127.0.0.1:8070`
- GROBID admin: `http://127.0.0.1:8071`
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
- `grobid`
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

That means:

- `/explore` can work once Qdrant contains indexed data
- `/classify` and `/precision-miner` still need an external Mongo instance via `MONGO_URI`
- GROBID-backed ingestion is available out of the box at `http://127.0.0.1:8070`
- the API container resolves GROBID internally via `GROBID_URL=http://grobid:8070`
- local CLI usage can point at the same container with `GROBID_URL=http://127.0.0.1:8070` and `--loader grobid`

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
- `tests/README.md`

## Current Status Notes

A few practical orientation notes:

- the FastAPI app, Streamlit UI, CLI, workflows, and `src/episcope` package structure are the clearest current entry points
- the `src/episcope/episcope.py` Typer CLI module is the installed `episcope` command
- the `notebooks/` folder contains an ordered learning path for local indexing, PDF ingestion, retrieval, and extraction workflows

When in doubt, prefer the package modules under `src/episcope/` and the tested imports in `tests/`.
