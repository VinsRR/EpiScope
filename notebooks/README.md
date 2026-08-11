# EpiScope Notebook Guide

These notebooks are a hands-on tour of EpiScope's **Python API**. They run
**offline and without any API key** — each workflow notebook uses a small
deterministic generator in place of a real LLM, and a tiny keyword embedder in
place of a real embedding model, so you can see the full
retrieval → prompt → parse → result loop without credentials, model downloads,
or network calls. Real usage would call `EmbedderFactory.get_embedder(...)`
instead, which defaults to the torch-free `fastembed` backend and downloads a
model on first use.

They can be run from either the repository root or the `notebooks/` directory.

## The `episcope_nb` Helper

Every notebook opens by importing [`episcope_nb.py`](episcope_nb.py), a small
module that sits next to the notebooks. It is **notebook scaffolding, not part of
the `episcope` package**, and it holds the two things every notebook would
otherwise copy:

- **Path bootstrap** — importing it adds `src/` to `sys.path` when it detects a
  local checkout; installed-package users need nothing.
- **Deterministic offline stand-ins** — `TinyKeywordEmbedder`,
  `FixedJSONGenerator`, `StaticRetriever`, two small paper corpora
  (`sample_papers()`, `classification_papers()`), and `build_index()` /
  `build_academic_db()` convenience constructors.

Notebooks still build things by hand where the construction *is* the lesson:
notebook 01 writes out `PaperMetadata` / `StructuredSection` objects, and
notebooks 02 and 04 wire up the indexer and retriever explicitly. The helper only
absorbs the boilerplate around those.

## Learning Path

1. `01_core_quickstart.ipynb`
   Build the smallest EpiScope loop: structured paper data, local indexing,
   retrieval, and provenance.

2. `02_indexing_and_retrieval.ipynb`
   Inspect chunking, vector-store payloads, corpus retrieval, paper-scoped
   retrieval, and payload filtering.

3. `03_pdf_ingestion.ipynb`
   Extract sections and metadata from the PDFs in `pdf_samples/` and store them
   in an `AcademicDB`.

4. `04_pdf_retrieval_and_workflows.ipynb`
   Index the sample PDFs, retrieve evidence, generate a trace, and run a
   structured extraction workflow.

5. `05_precision_miner_workflow.ipynb`
   Run `PrecisionMiner` on controlled evidence and inspect the detailed
   workflow trace.

6. `06_data_source_extraction.ipynb`
   Validate data-source extraction JSON, review whether extracted items are
   supported by retrieved evidence, then define a **new miner declaratively**
   with a `TaskSpec`.

7. `07_classification_workflow.ipynb`
   Run a built-in classification task, then **define your own task two ways**:
   declaratively with a `TaskSpec` (the recommended path), and by building a
   low-level `BaseClassifierConfig` by hand.

8. `08_service_runtime.ipynb`
   Step up from hand-composed objects to `EpiScopeRuntime` — the service layer
   the CLI, API, and UI all share — including its local-first fallback and the
   `explore` / `classify` / `precision_mine` verbs.

9. `09_workspaces.ipynb`
   Configuration-as-a-folder: create a workspace, watch auto-discovery walk up
   for `episcope.toml`, drive the runtime from it, and drop a declarative task
   into `tasks/`.

## Beyond The Notebooks

The notebooks focus on the Python API. EpiScope also offers higher-level entry
points that are documented in the README and the
[project wiki](https://github.com/VinsRR/EpiScope/wiki):

- **CLI** — `episcope quickstart`, `doctor`, `init`, `index`, `inspect`,
  `papers`, `explore`, `ask`, `classify`, `precision-miner`, `tasks`, `serve`,
  and `studio`. Start with
  [Local CLI Quickstart](https://github.com/VinsRR/EpiScope/wiki/Local-CLI-Quickstart).
- **Workspaces** — `episcope init <name>` creates a self-contained folder
  (`episcope.toml`, `papers/`, `index/`, `outputs/`, `tasks/`) that commands
  auto-discover when run from inside it. Notebook 09 covers the Python side; see
  [Workspaces](https://github.com/VinsRR/EpiScope/wiki/Workspaces) for the CLI
  walkthrough.
- **Declarative tasks** — define classifiers/miners as JSON, scaffold them with
  `episcope tasks new` (add `--interactive` for a guided wizard), validate with
  `episcope tasks validate`, and run them from the CLI (`--task-file`) or
  inline via the API. See
  [Declarative Tasks](https://github.com/VinsRR/EpiScope/wiki/Declarative-Tasks).
- **UI** — `episcope studio` runs the Streamlit interface local-first, with no
  external services required. `episcope serve` plus the UI over a shared
  Qdrant/MongoDB corpus is the deployed/multi-user setup. See
  [Streamlit UI](https://github.com/VinsRR/EpiScope/wiki/Streamlit-UI) and
  [Server Mode and Docker](https://github.com/VinsRR/EpiScope/wiki/Server-Mode-and-Docker).

## Running The Notebooks

Install the package (and Jupyter), then open the notebooks:

```bash
python -m pip install "epi-scope @ git+https://github.com/VinsRR/EpiScope.git@main"
python -m pip install jupyterlab
jupyter lab notebooks/
```

Run cells top to bottom. To use a real model instead of the deterministic demo
generators, replace the demo `Generator` with `episcope.rag.generation.llm_generator.LLMGenerator`
and set the relevant provider key (`GEMINI_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `OLLAMA_HOST`).

One behaviour changes when you do: classifier and miner configs default to
`structured_output="schema"`, so the workflow passes a `response_schema` down to
the generator and the provider is *constrained* at decode time to emit conforming
JSON, rather than merely asked to. The demo generators ignore that kwarg. Set
`structured_output="json"` or `"off"` on the config (or `--structured-output` on
the CLI) if a provider or model does not support it.

## Sample PDFs

`pdf_samples/` contains small open-access PDFs used by the ingestion and PDF
retrieval notebooks. PDF parsing can take a minute on the first run because the
parser may initialize local caches.

Which parser actually runs depends on what you installed, and the two produce
noticeably different section structure:

- **Base install** — `unstructured`'s hi_res PDF path is unavailable, so the
  loader falls back to a torch-free `pdfminer.six` heuristic. It is fast and
  dependency-light, but section titles are rough.
- **`pip install "epi-scope[local-ml]"`** — adds `unstructured[pdf]`, enabling
  layout-model parsing with better section boundaries (and a much heavier
  install).
- **GROBID** — for the cleanest structure, point EpiScope at a running GROBID
  service with `--loader grobid` (see the wiki Troubleshooting page).

Notebooks 03 and 04 print section counts and titles, so expect those numbers to
differ between the first two options.
