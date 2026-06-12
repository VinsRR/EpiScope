# EpiScope Notebook Guide

These notebooks are a hands-on tour of EpiScope's **Python API**. They run
**offline and without any API key** — each workflow notebook uses a small
deterministic generator in place of a real LLM, so you can see the full
retrieval → prompt → parse → result loop without credentials or network calls.

They can be run from either the repository root or the `notebooks/` directory.
When a notebook detects a local checkout it adds `src/` to `sys.path`;
installed-package users do not need that step.

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
   Validate data-source extraction JSON and review whether extracted items are
   supported by retrieved evidence.

7. `07_classification_workflow.ipynb`
   Run a built-in classification task, then **define your own task two ways**:
   declaratively with a `TaskSpec` (the recommended path), and by building a
   low-level `BaseClassifierConfig` by hand.

## Beyond The Notebooks

The notebooks focus on the Python API. EpiScope also offers higher-level entry
points that are documented in the README and the
[project wiki](https://github.com/VinsRR/EpiScope/wiki):

- **CLI** — `episcope doctor`, `index`, `explore`, `ask`, `classify`,
  `precision-miner`, and `tasks`. Start with
  [Local CLI Quickstart](https://github.com/VinsRR/EpiScope/wiki/Local-CLI-Quickstart).
- **Declarative tasks** — define classifiers/miners as JSON, scaffold them with
  `episcope tasks new`, validate with `episcope tasks validate`, and run them
  from the CLI (`--task-file`) or inline via the API. See
  [Declarative Tasks](https://github.com/VinsRR/EpiScope/wiki/Declarative-Tasks).
- **Server + UI** — a FastAPI backend and Streamlit interface over a shared
  Qdrant/MongoDB corpus. See
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
and set the relevant provider key (for example `GEMINI_API_KEY`).

## Sample PDFs

`pdf_samples/` contains small open-access PDFs used by the ingestion and PDF
retrieval notebooks. PDF parsing can take a minute on the first run because the
parser may initialize local caches. The default `unstructured` loader is
dependency-light but produces rough section titles; for cleaner structure point
EpiScope at a running GROBID service (see the wiki Troubleshooting page).
