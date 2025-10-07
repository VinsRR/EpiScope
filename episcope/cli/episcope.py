"""
cli/episcope.py

Command line interface for EpiScope using Typer.  This CLI exposes
commands to ingest documents and query the system.  It is intended for
local experimentation and development; for production use the REST API
may be more appropriate.
"""
from __future__ import annotations

import typer
from typing import Optional

from ..ingest.pipeline import IngestPipeline
from ..generate.answer import AnswerGenerator

app = typer.Typer(help="EpiScope CLI")


@app.command()
def ingest(
    file_path: Optional[str] = typer.Option(None, help="Path to a PDF to ingest"),
    directory: Optional[str] = typer.Option(None, help="Directory containing PDFs to ingest"),
    doi: Optional[str] = typer.Option(None, help="DOI of a document to ingest"),
    rag_method: str = typer.Option("text", help="RAG method to use (default: text)"),
):
    """Ingest a PDF file, directory of PDFs or a DOI."""
    pipeline = IngestPipeline(rag_method)
    if file_path:
        pipeline.ingest_pdf(file_path)
    elif directory:
        pipeline.ingest_directory(directory)
    elif doi:
        pipeline.ingest_doi(doi)
    else:
        typer.echo("Provide either --file-path, --directory or --doi")


@app.command()
def query(
    query: str = typer.Argument(..., help="Question to answer"),
    rag_method: str = typer.Option("text", help="RAG method to use"),
    top_k: int = typer.Option(5, help="Number of contexts to retrieve"),
):
    """Query the system and print the answer with provenance."""
    generator = AnswerGenerator(rag_method)
    provenance = generator.answer_question(query, top_k=top_k)
    typer.echo(provenance.answer)
    for ev in provenance.evidences:
        typer.echo(f"- {ev.paper_id} [{ev.section}]: {ev.snippet[:100]}...")


if __name__ == "__main__":
    app()
