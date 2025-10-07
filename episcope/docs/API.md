# API Reference

EpiScope exposes a minimal REST API via FastAPI.  The API is defined
in `src/api.py` and serves two primary endpoints: `/ingest` and
`/query`.

## POST `/ingest`

Ingests a document into the system.  Accepts one of the following
properties in the JSON body:

* `file_path` – Path to a PDF file on the server to ingest.
* `directory` – Path to a directory containing PDFs to ingest
  recursively.
* `doi` – A DOI string to ingest.  (Not yet implemented.)
* `rag_method` – Name of the RAG method to use for indexing.  Default
  is `text`.

Returns a JSON object with a status field.  On error returns a 400
response.

Example:

```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"file_path": "/data/papers/influenza.pdf", "rag_method": "text"}'
```

## POST `/query`

Queries the system and returns an answer with provenance.  Expects the
following properties in the JSON body:

* `query` – The user question.
* `rag_method` – Name of the RAG method to use.  Default is `text`.
* `top_k` – Number of contexts to retrieve (default 5).

Returns a JSON object with two fields:

* `answer` – The generated answer as a string.
* `evidences` – A list of evidence objects.  Each evidence contains
  `paper_id`, `snippet`, `section`, `index_version`, `model_id` and
  `prompt_id`.

Example:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How long is the incubation period for influenza?", "top_k": 3}'
```

Response:

```json
{
  "answer": "The incubation period for influenza is typically 1–4 days.",
  "evidences": [
    {
      "paper_id": "2020_Lin_r0_basic__structured_chunks.json",
      "snippet": "... the incubation period for influenza A (H1N1) is around 2 days ...",
      "section": "Abstract",
      "index_version": null,
      "model_id": "llama3:8b",
      "prompt_id": "text"
    },
    { ... }
  ]
}
```

## Running the API

To start the API locally:

```bash
uvicorn episcope.src.api:app --reload --port 8000
```

The API automatically reloads on code changes when run with the
`--reload` flag.
