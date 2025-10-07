# Configuration Guide

EpiScope is designed to be highly configurable.  The configuration
system is based on [Pydantic](https://docs.pydantic.dev/) and can be
overridden via environment variables or customised configuration
files.

## Global Settings (`settings.py`)

The `settings.py` module defines three Pydantic settings classes:

### `ProviderConfig`

Controls which language model provider and model are used.  The
following fields are available:

| Field    | Description | Environment Variable |
|---------|-------------|----------------------|
| provider | Name of the LLM provider.  One of `local`, `openai`, `gemini`. | `EPISCOPE_PROVIDER_PROVIDER` |
| model    | Model identifier to use with the selected provider.  For the local provider this is the Ollama model name (e.g. `llama3:8b`). | `EPISCOPE_PROVIDER_MODEL` |
| api_key | API key for proprietary providers (OpenAI or Gemini). | `EPISCOPE_PROVIDER_API_KEY` |

### `IndexConfig`

Configures the Qdrant vector index used by the Explorer pipeline.

| Field          | Description | Environment Variable |
|---------------|-------------|----------------------|
| collection_name | Name of the Qdrant collection. | `EPISCOPE_INDEX_COLLECTION_NAME` |
| url            | Base URL for the Qdrant service. | `EPISCOPE_INDEX_URL` |
| timeout        | Timeout (in seconds) for Qdrant requests. | `EPISCOPE_INDEX_TIMEOUT` |
| distance       | Vector distance metric (`COSINE`, `EUCLID`, `DOT`, `MANHATTAN`). | `EPISCOPE_INDEX_DISTANCE` |

### `AppConfig`

Aggregates `ProviderConfig` and `IndexConfig`.  A singleton instance
(`CONFIG`) is created on import and should be used throughout the code
base to access current settings.

## RAG Configs

The default behaviour of the Explorer pipeline is governed by the
configuration modules in `configs/`.  There are separate modules for
each retrieval strategy:

* `text_only.py` – defines the embedding models, batch sizes, Qdrant
  collection name and prompt templates for the text‑only RAG.  The
  defaults are suitable for a local Ollama and Qdrant setup.
* `ColPali.py` – configuration for the ColPali (multimodal) RAG.
* `graph.py` – configuration for the GraphRAG pipeline.
* `CLIP.py` – configuration for multimodal embedding via CLIP.

These config files can be edited to change the default embedding
models or collection names.  Alternatively, you can pass keyword
arguments into `RAGFactory.get()` to override specific settings at
runtime.

## PrecisionMiner Configs

The PrecisionMiner pipeline is configured via the dataclasses and
constants defined in `parse/configs/configs.py`.  Key parameters
include:

* `SearchConfig` – controls query generation and search parameters for
  the structured chunk querying.  Fields include `top_k` and
  `similarity_threshold`.
* `PipelineConfig` – controls the parallelism and timeouts for the
  processing pipeline.  Fields include `max_workers` and
  `include_tables`.
* `GT_PAPER_TYPES` and `SELECTED_TYPE` – specify which paper types to
  process during ingestion and extraction.

You can customise these values by importing and modifying the
classes, or by extending them with additional fields.
