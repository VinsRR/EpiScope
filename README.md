# EpiLens

EpiLens is a local-first Python toolkit for evidence-grounded analysis of
scientific literature. It parses and indexes papers on your machine, retrieves
the passages relevant to a question, and can use a local or hosted language
model to produce answers and structured outputs with inspectable provenance.

It is designed for epidemiological and public-health literature, but its
retrieval, classification, and extraction workflows can be adapted to other
scientific domains. The distribution name, import package, and command are all
`epilens`.

## Install

EpiLens requires Python 3.10 or newer:

```bash
python -m pip install epilens
epilens --version
```

The base install includes lightweight PDF parsing and local semantic retrieval.
It does not install every model-provider SDK, a database server, or the larger
ML parsing stack. Add only what you plan to use; the [extras table](#optional-extras)
shows the choices.

## A Five-Minute First Run

This path starts with no API key, database, Docker service, or configuration
file. Replace `paper.pdf` with one of your own papers.

### 1. Inspect a paper

```bash
epilens inspect paper.pdf
```

This uses the included `pdfminer` parser. It verifies that the document is
readable and reports the detected sections and metadata. It works offline and
does not download a model.

### 2. Retrieve relevant evidence

```bash
epilens explore "What data sources were used?" --path paper.pdf --quality fast
```

The first semantic-retrieval command downloads the small
`sentence-transformers/all-MiniLM-L6-v2` embedding model (about 90 MB) and
caches it for later runs. No LLM key is needed; the command returns the most
relevant source passages.

### 3. Generate an answer when you are ready

Without a configured LLM, `ask` still returns the retrieved passages instead of
failing:

```bash
epilens ask "What data sources were used?" --path paper.pdf --quality fast
```

To generate a synthesized answer, install one provider and run the guided
setup. For example, with Gemini:

```bash
python -m pip install "epilens[gemini]"
epilens quickstart
epilens ask "What data sources were used?" --path paper.pdf --quality fast
```

`quickstart` writes the selected provider and key to a local, ignored `.env`
file. It does not contact optional services unless you pass `--probe`. You can
also use local Ollama without a provider SDK or API key.

### 4. Keep a reusable literature workspace

For more than one command or paper, create a workspace:

```bash
epilens init my-review
epilens index ./papers --workspace my-review
epilens papers --workspace my-review
epilens ask "Which papers mention GenBank?" --workspace my-review
```

The workspace is an ordinary directory containing `epilens.toml`, copied
papers, a local vector index, metadata, task definitions, outputs, and logs.
Run commands from inside it, or pass `--workspace` from anywhere.

## What EpiLens Does

- Parses PDFs, text, and Markdown with a deterministic lightweight default.
- Indexes one paper or a corpus into a local file store without a database
  server.
- Retrieves semantically relevant passages and keeps their paper/section
  provenance.
- Answers questions over retrieved evidence using Gemini, OpenAI, OpenRouter,
  Anthropic, or Ollama.
- Runs schema-validated classification across analyst-defined label axes.
- Extracts structured items such as data sources, supplementary links, and key
  references.
- Lets researchers define new classifiers and extractors as JSON task specs,
  without changing Python code.
- Scales to optional FAISS or Qdrant vector stores, MongoDB metadata, GROBID
  parsing, a FastAPI service, and a Streamlit Studio.

## Mental Model

EpiLens follows the three-stage workflow described in the accompanying paper:

1. **Ingest and index:** parse papers, split them into chunks, embed the chunks,
   and store them locally.
2. **Retrieve:** find the most relevant chunks for a query or fixed workflow
   template, optionally filtering by paper or section.
3. **Generate and validate:** pass only the selected evidence to an LLM,
   validate structured outputs against a schema, and retain provenance.

The LLM is a controlled linguistic step, not a knowledge database. Corpus
storage, retrieval, task logic, and validation remain local; only the selected
evidence is sent to a remote provider when you choose one.

## Main Commands

| Command | Purpose | Needs an LLM? |
| --- | --- | --- |
| `epilens inspect FILE` | Parse and summarize a document | No |
| `epilens explore QUERY --path PATH` | Retrieve matching passages | No |
| `epilens ask QUERY --path PATH` | Retrieve and optionally synthesize | Optional |
| `epilens init DIR` | Create a reusable workspace | No |
| `epilens index PATH` | Add papers to a local index | No |
| `epilens papers` | List indexed papers | No |
| `epilens classify --file FILE` | Apply a classification workflow | Yes |
| `epilens precision-miner --file FILE` | Extract structured items | Yes |
| `epilens tasks` | List built-in and user-defined tasks | No |
| `epilens doctor` | Check local essentials without network probes | No |
| `epilens studio` | Start the API and browser UI together | Depends on action |

Run `epilens COMMAND --help` for examples and advanced controls. `--quality
fast|balanced|accurate` is the simplest way to tune chunking and retrieval;
individual settings remain available for experienced users.

## Optional Extras

Install extras with `python -m pip install "epilens[EXTRA]"`.

| Extra | Adds |
| --- | --- |
| `gemini` | Google Gemini SDK |
| `openai` | OpenAI SDK |
| `openrouter` | OpenAI-compatible SDK used for OpenRouter |
| `anthropic` | Anthropic SDK |
| `providers` | All hosted-provider SDKs |
| `server` | FastAPI, Uvicorn, and uploads |
| `ui` | Streamlit and pandas |
| `all` | Local file-backed API + Studio UI (`server,ui`) |
| `faiss` | Local FAISS vector backend |
| `qdrant` | Qdrant client; a Qdrant server is still required |
| `mongo` | MongoDB client; a MongoDB server is still required |
| `grobid` | GROBID client and XML parser; a GROBID server is still required |
| `local-ml` | Torch-based embeddings and Unstructured `hi_res` PDF parsing |
| `dev` | Tests, lint, type checking, and build/release tools |

Examples:

```bash
python -m pip install "epilens[openai]"      # one hosted provider
python -m pip install "epilens[all,gemini]" # Studio plus Gemini
python -m pip install "epilens[grobid]"     # richer academic-PDF parsing
```

## Configuration

EpiLens loads environment variables from your shell and from the nearest `.env`
file. Common settings are:

```dotenv
EPILENS_LLM_PROVIDER=gemini
EPILENS_LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-key-here

# Optional local overrides
EPILENS_EMBED_PROVIDER=auto
EPILENS_DEVICE=cpu
EPILENS_OUTPUT_FORMAT=human
```

Other provider credentials are `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and
`ANTHROPIC_API_KEY`; Ollama uses `OLLAMA_HOST`. External deployments can set
`QDRANT_URL`, `QDRANT_COLLECTION`, `MONGO_URI`, `MONGO_DB_NAME`, and
`GROBID_URL`. The complete annotated template is
[`.env.example`](https://github.com/VinsRR/EpiScope/blob/main/.env.example).

Secrets belong in `.env` or the deployment secret store, never in a workspace
file, notebook, issue, or commit.

## Structured Workflows

The built-in workflow families are:

- `PaperClassifier`, for evidence-backed labels such as paper type,
  geographical coverage, data type, and reported data accessibility.
- `PrecisionMiner`, for variable-length structured extraction such as data
  sources, supplementary material, and key references.

A workflow fixes two task-specific artifacts across every paper: retrieval and
prompt instructions, plus a Pydantic/JSON output schema. Reusing the same
template and validating the same schema makes corpus-level outputs comparable
and machine-readable.

Create a task interactively:

```bash
epilens tasks new --kind classifier --interactive
epilens tasks new --kind miner --interactive
```

Place the resulting JSON file in a workspace's `tasks/` directory for automatic
discovery, pass it directly with `--task-file`, or expose a directory through
`EPILENS_TASKS_DIR`.

## Python API

Start at the public package surface:

```python
from epilens import EpiLensRuntime, PaperClassifier, PrecisionMiner

runtime = EpiLensRuntime()
health = runtime.health()
print(health.checks)
```

For a small, fully local parsing example:

```python
from epilens.rag.ingestion.document_loader import DocumentLoaderFactory

loader = DocumentLoaderFactory.get_loader("pdfminer")
sections, metadata, references = loader.load("paper.pdf")

print(metadata.title)
print(sections[0].content[:500])
```

The ordered notebooks progressively introduce schemas, indexing, retrieval,
PDF ingestion, structured extraction/classification, the runtime facade, and
workspaces. They use helper files and sample PDFs that are intentionally not
bundled in the wheel, so clone the repository before running them:

```bash
git clone https://github.com/VinsRR/EpiScope.git
cd EpiScope
python -m pip install -e ".[dev]"
jupyter lab notebooks/
```

See the
[notebook guide](https://github.com/VinsRR/EpiScope/tree/main/notebooks) for the
recommended order.

## Studio and API

The easiest browser path uses local file-backed workspaces and requires no
Qdrant or MongoDB:

```bash
python -m pip install "epilens[all]"
epilens studio
```

Studio starts FastAPI and Streamlit together on `127.0.0.1`, manages both
processes, and opens the workspace UI. To run the parts separately:

```bash
epilens serve --port 8000
epilens-ui
```

The API includes health, retrieval, classification, precision-mining, and
workspace-scoped routes. Interactive API documentation is available at
`http://127.0.0.1:8000/docs` while the server is running.

For a shared deployment with Qdrant and GROBID, use the repository's
[Docker Compose configuration](https://github.com/VinsRR/EpiScope/blob/main/docker-compose.yml).
MongoDB remains external and is only needed for the legacy unscoped
corpus-backed routes; the workspace-scoped Studio path is file-backed.

## Name Migration

The PyPI distribution, Python namespace, and executable are now `epilens`.
There is intentionally no `episcope` import or command shim because that name
belongs to an unrelated package on PyPI.

To avoid hiding existing local data during the rename, EpiLens still reads old
`EPISCOPE_*` environment variables when the equivalent `EPILENS_*` value is not
set, discovers existing `episcope.toml` workspaces, and reuses an existing
`.episcope/` state directory when no `.epilens/` directory exists. New files and
configuration use the EpiLens names.

The default external MongoDB database and Qdrant collection are now
`epilens_academic_db` and `epilens_academic`. Set `MONGO_DB_NAME` and
`QDRANT_COLLECTION` explicitly if you want to keep using stores created under
older defaults.

## Development

```bash
git clone https://github.com/VinsRR/EpiScope.git
cd EpiScope
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all,qdrant,mongo,faiss,providers]"
pytest -q
ruff check src/epilens tests
python -m build
python -m twine check dist/*
```

CI tests the base package across Python 3.10-3.14 on Linux and representative
macOS and Windows versions, exercises optional API/UI/store/provider paths,
validates both distribution artifacts, and installs the wheel in an isolated
environment outside the source tree.

Releases use PyPI Trusted Publishing. Maintainers configure the GitHub `pypi`
environment once, update the version and changelog, then push a matching tag
such as `v0.1.0`; the release workflow rejects mismatched tags before upload.

## Citation

The software and accompanying manuscript were developed by Vincenzo Perri,
Ciro Cattuto, and Daniela Paolotti at ISI Foundation. Citation metadata is in
[`CITATION.cff`](https://github.com/VinsRR/EpiScope/blob/main/CITATION.cff).
The manuscript is still a draft, so a journal reference and DOI should be added
there when available.

## License

EpiLens is distributed under the
[GNU Affero General Public License v3.0](https://github.com/VinsRR/EpiScope/blob/main/LICENSE.txt)
(`AGPL-3.0-only`).
