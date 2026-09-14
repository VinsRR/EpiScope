# Changelog

All notable user-facing changes are recorded here. This project follows
[Semantic Versioning](https://semver.org/) once releases are published.

## 0.1.0 - Unreleased

### Added

- A local-first CLI for document inspection, indexing, retrieval, question
  answering, workspaces, classification, and structured extraction.
- File-backed defaults plus optional FAISS, Qdrant, MongoDB, GROBID, FastAPI,
  Streamlit, hosted-provider, and local-ML integrations.
- PyPI metadata, typed-package marker, isolated artifact checks, and a Trusted
  Publishing release workflow.
- A deterministic `pdfminer` parser option and default that needs no model or
  external service.

### Changed

- Renamed the distribution, Python import, and command to `epilens`.
- Split heavyweight parsers, databases, and model-provider SDKs into explicit
  extras so the default install stays focused.
- Reworked onboarding around an install-first, no-key learning path.
