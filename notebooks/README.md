# EpiScope Notebook Guide

The notebooks are ordered as a short learning path. They can be run from either
the repository root or the `notebooks/` directory. When a notebook detects a
local checkout, it adds `src/` to `sys.path`; installed-package users do not
need that step.

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
   Run a built-in classification task and define a custom category set for a
   new paper-classification task.

## Sample PDFs

`pdf_samples/` contains small PDFs used by the ingestion and PDF retrieval
notebooks. PDF parsing can take a minute on the first run because parser
dependencies may initialize local caches.
