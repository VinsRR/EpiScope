import typer
from typing_extensions import Annotated
import logging

from episcope.rag.ingestion.document_loader import DocumentLoaderFactory
from episcope.db.db_factory import get_academic_db
from episcope.pipelines.precision_miner_pipeline import PipelineProcessor

app = typer.Typer()

@app.command()
def run_pipeline(
    input_dir: Annotated[str, typer.Option(help="Directory containing PDFs to process.")],
    strategy_name: Annotated[str, typer.Option(help="Name of the extraction strategy.")],
    loader: Annotated[str, typer.Option(help="Document loader to use ('grobid' or 'unstructured').")] = "grobid",
    db_uri: Annotated[str, typer.Option(help="MongoDB connection URI.")] = "mongodb://localhost:27017",
    db_name: Annotated[str, typer.Option(help="Name of the MongoDB database.")] = "AcademicCorpus",
    use_in_memory_db: Annotated[bool, typer.Option(help="Use an in-memory database instead of MongoDB.")] = False,
):
    """
    Run the full PrecisionMiner pipeline on a directory of PDFs.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Set up database
    db = get_academic_db(uri=db_uri, db_name=db_name, use_in_memory=use_in_memory)
    logger.info(f"Using {'in-memory' if use_in_memory else 'MongoDB'} database.")

    # 2. Ingest documents
    logger.info(f"Starting ingestion from '{input_dir}' with loader '{loader}'...")
    doc_loader = DocumentLoaderFactory.get_loader(loader)
    doc_loader.extract_directory(input_dir, strategy_name=strategy_name, db=db)
    logger.info("Ingestion complete.")

    # 3. Run PrecisionMiner pipeline
    logger.info("Starting PrecisionMiner pipeline...")
    pipeline = PipelineProcessor(db=db)
    pipeline.run(strategy_name=strategy_name)
    logger.info("PrecisionMiner pipeline complete.")

if __name__ == "__main__":
    app()
