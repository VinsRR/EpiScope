import os
import shutil
from pathlib import Path
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
import asyncio
from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingfaceEmbeddings

# Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# Ensure episcope is importable. Assumes it's installed in editable mode.
# try:
from episcope.vectordb.faiss import FaissDB
from episcope.rag.embeddings.huggingface import HuggingFaceEmbedder
from episcope.rag.indexing.indexer import Indexer
from episcope.rag.indexing.chunking import ParagraphChunker
from episcope.rag.retrieval.semantic import SemanticRetriever
from episcope.rag.generation.llm_generator import LLMGenerator
from episcope.clients import OllamaClient
from episcope.schemas import PaperMetadata, StructuredSection


# --- 1. Setup evaluation data ---
CORPUS = {
    "R0": "The basic reproduction number, R0, of an infection is the expected number of secondary cases produced by a single infection in a completely susceptible population. For influenza, the R0 is estimated to be between 0.9 and 2.1.",
    "IncubationPeriod": "The incubation period of a disease is the time from exposure to the infectious agent to the onset of symptoms. For COVID-19, it is typically 5-6 days, but can range from 2 to 14 days.",
    "HerdImmunity": "Herd immunity is a form of indirect protection from infectious disease that occurs when a sufficient percentage of a population has become immune to an infection, whether through vaccination or previous infections, thereby reducing the likelihood of infection for individuals who lack immunity."
}
EVAL_QUESTIONS = [
    "What is the basic reproduction number?",
    "What is the incubation period for COVID-19?",
    "How is herd immunity achieved?",
]
EVAL_GROUND_TRUTHS = [
    "The basic reproduction number (R0) is the expected number of secondary cases produced by a single infection in a completely susceptible population.",
    "The incubation period for COVID-19 is typically 5-6 days, ranging from 2 to 14 days.",
    "Herd immunity can be achieved through vaccination or previous infections.",
]

# Setup RAG pipeline: using episcope components
def setup_rag_pipeline():
    """Sets up the RAG pipeline using episcope components."""
    print("Setting up RAG pipeline...")
    index_dir = Path("./ragas_evaluation/faiss_index")
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    # Use a small, fast embedding model for this example
    embedder = HuggingFaceEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    db = FaissDB(index_dir=str(index_dir))
    chunker = ParagraphChunker(min_chunk_size=5)
    indexer = Indexer(db=db, embedder=embedder, chunker=chunker)

    print("Indexing documents...")
    for doc_id, content in CORPUS.items():
        sections = [StructuredSection(content=content)]
        metadata = PaperMetadata(title=doc_id)
        indexer.index_paper(sections, metadata, paper_id=doc_id)
    
    db.save()

    retriever = SemanticRetriever(vectordb=db)
    client = OllamaClient()
    generator = LLMGenerator(client=client, model="llama3.2:latest")

    return retriever, generator

# --- 3. Run pipeline and collect data ---
def run_pipeline_and_collect_data(retriever, generator):
    """Runs EpiScope pipeline for each question and collects the results."""
    print("Running pipeline to collect answers and contexts...")
    results = []
    for i, question in enumerate(EVAL_QUESTIONS):
        contexts = retriever.retrieve(question, top_k=2)
        context_texts = [ctx.text for ctx in contexts]
        
        provenance = generator.generate(contexts=contexts, question=question)
        answer = provenance.answer

        results.append({
            "question": question,
            "answer": answer,
            "contexts": context_texts,
            "ground_truth": EVAL_GROUND_TRUTHS[i]
        })
    return results

# --- Main execution ---

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def main():
    pipeline = setup_rag_pipeline()
    if not pipeline:
        return

    retriever, generator = pipeline
    evaluation_data = run_pipeline_and_collect_data(retriever, generator)

    print("\n\n--- Evaluation Data ---")
    for item in evaluation_data:
        print(item)
    print("-------------------------\n")
    
    # --- 4. Evaluate with RAGAS ---
    print("Evaluating with RAGAS...")
    
    dataset = Dataset.from_list(evaluation_data)
    # The critic validates the generated answers (based on contexts) against the ground truth answers
    critic_llm = Ollama(model="qwen2.5vl:3b")
    # needs an embedding model because several metrics (like 'answer relevancy' or 'context precision')
    # rely on calculating the semantic similarity between text segments
    ollama_emb = OllamaEmbeddings(model="nomic-embed-text")



    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=critic_llm,
        embeddings=ollama_emb,
    )
    
    print("\n--- RAGAS Evaluation Results ---")
    print(result)
    print("---------------------------------")

if __name__ == "__main__":
    main()