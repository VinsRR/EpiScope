"""
src/api.py

HTTP API for EpiScope.

This module defines a FastAPI application exposing endpoints for
ingestion and querying of the EpiScope system.  The API is intentionally
minimal: it provides a way to ingest documents into the system and to
retrieve answers with provenance.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from episcope.rag.ingestion.pipeline import IngestPipeline
from episcope.rag.generation.answer import AnswerGenerator

app = FastAPI(title="EpiScope API")


class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    directory: Optional[str] = None
    doi: Optional[str] = None
    rag_method: str = "text"


class QueryRequest(BaseModel):
    query: str
    rag_method: str = "text"
    top_k: int = 5


class EvidenceResponse(BaseModel):
    paper_id: str
    snippet: str
    section: Optional[str] = None
    index_version: Optional[str] = None
    model_id: Optional[str] = None
    prompt_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    evidences: list[EvidenceResponse]


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, str]:
    pipeline = IngestPipeline(rag_method=request.rag_method)
    if request.file_path:
        pipeline.ingest_pdf(request.file_path)
    elif request.directory:
        pipeline.ingest_directory(request.directory)
    elif request.doi:
        pipeline.ingest_doi(request.doi)
    else:
        raise HTTPException(status_code=400, detail="Must provide file_path, directory or doi")
    return {"status": "ok"}


@app.post("/query", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    generator = AnswerGenerator(rag_method=request.rag_method)
    provenance = generator.answer_question(request.query, top_k=request.top_k)
    return AnswerResponse(
        answer=provenance.answer,
        evidences=[
            EvidenceResponse(
                paper_id=e.paper_id,
                snippet=e.snippet,
                section=e.section,
                index_version=e.index_version,
                model_id=e.model_id,
                prompt_id=e.prompt_id,
            )
            for e in provenance.evidences
        ],
    )

