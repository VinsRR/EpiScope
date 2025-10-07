# Migration Notes

This document describes how the original repository has been refactored
into the new EpiScope structure.  It serves as a guide for developers
transitioning from the old codebase to the refactored one.

## Original Layout

The archive contained three top‑level packages:

* `rag_tool` – a collection of scripts implementing various
  retrieval‑augmented generation pipelines and a Streamlit UI.
* `source_extraction` – an early prototype for GROBID parsing and
  table extraction.
* `source_extraction_mod` – a more complete implementation of the
  PrecisionMiner pipeline, including data models, extractors,
  indexing and querying modules.

## New Layout

The new repository is structured as follows:

* `configs/` – contains the configuration modules from `rag_tool/configs`.
* `core/` – defines abstract base classes for RAG methods (`core/rag.py`),
  LLM providers (`core/provider.py`) and provenance models
  (`core/provenance.py`).
* `ingest/` – provides a high‑level ingestion pipeline (`ingest/pipeline.py`)
  that wraps the RAG indexer.  The old helper scripts (e.g.
  `get_pdf_functions.py`) have not yet been ported.
* `parse/` – contains the entire `source_extraction_mod` package,
  repackaged under `parse/` for clarity.  The modules remain largely
  unchanged and provide the PrecisionMiner functionality.
* `index/` – introduces `index/unified_db.py`, a refactored wrapper
  around the Qdrant client replacing `QdrantUnifiedDB` from
  `rag_tool/utils.py`.
* `retrieve/` – hosts the RAG implementations.  The text‑only pipeline
  has been ported to `retrieve/rag/text.py`, while the ColPali and
  GraphRAG pipelines remain TODO.  A factory class lives in
  `retrieve/rag/factory.py`.
* `generate/` – adds the `AnswerGenerator` class which orchestrates
  retrieval, LLM invocation and provenance construction.
* `providers/` – provides pluggable LLM providers for local
  (Ollama), OpenAI and Gemini.  These correspond to the various ways
  `ollama.chat` was invoked in the old code.
* `src/` – contains a FastAPI application exposing `/ingest` and
  `/query` endpoints.  This replaces the Streamlit app as the primary
  service entrypoint.
* `ui/` – retains a simple Streamlit front‑end (`ui/streamlit_app.py`).
* `cli/` – adds a Typer‑based CLI for ingestion and querying.
* `docs/` – contains documentation files (architecture, configs,
  evaluation protocol, API reference).

## Removed Components

Some parts of the original repository have not yet been brought over:

* The PDF downloading utilities (`get_pdf_functions.py`,
  `pdf_utils.py`, `pubmed_functions.py`) are not currently used.  They
  should be ported into the ingestion pipeline in a future revision.
* The ColPali and Graph RAG implementations from `rag_tool` still exist
  in the archive but have not yet been refactored.  They can be
  ported into `retrieve/rag/` by following the pattern used for
  `TextRAG`.
* The `source_extraction/pipeline_rag.py` prototype has been
  superseded by the full implementation in `parse/`.

