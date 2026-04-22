# TableRef Extractor

`tableref` is a Python package for extracting tables from scientific papers (PDFs) and matching bibliographic references found within those tables.

This tool is designed to help researchers and developers working on meta-analysis, knowledge extraction, and building literature-based knowledge graphs. It provides a powerful pipeline to turn unstructured tables in PDFs into structured, linked data.

## Features

- **Multi-Strategy Table Extraction**: Uses several backends (including `gmft`, `img2table`, and vision LLMs) to robustly extract tables from PDFs.
- **Advanced Reference Matching**: Employs both lightweight fuzzy matching and a sophisticated, ColBERT-inspired late-interaction model using Transformers for high-accuracy reference matching.
- **DOI Enrichment**: Fetches DOIs and other metadata from Crossref to enrich and standardize matched references.
- **Modular and Extensible**: Designed to be used as a standalone library.

## Installation

You can install the package and its dependencies using pip:

```bash
pip install .
```

For the full feature set, including the Transformer-based matcher, install the `torch` extra:

```bash
pip install .[torch]
```
