# configs/multimodal_config.py

# Embedding
# On how to choose the best model, see: https://unstructured.io/blog/understanding-embedding-models-make-an-informed-choice-for-your-rag
# On chucking for RAG: https://unstructured.io/blog/chunking-for-rag-best-practices
EMBED_MODEL  =  "Qwen/Qwen3-Embedding-0.6B"  #"nomic-ai/nomic-embed-text-v1.5"
SPARSE_EMBED_MODEL = "Qdrant/bm25"  # For sparse embeddings, e.g., for BM25
LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"

GENERATOR_MODEL = "qwen3:8b"  # "deepseek-r1:7b"# "llama3.2-vision" # 


BATCH_SIZE   = 4




# Qdrant
QDRANT_URL     = "http://localhost:6334" #"http://qdrant:6334" #
QDRANT_TIMEOUT = 60
BINARY_QUANTIZATION = True  # Whether to use binary quantization for embeddings

# Collection
COLLECTION_NAME = "test_text" #"multimodal_collection"
# VECTOR_SIZE     = 768
DISTANCE        = "COSINE"






SYSTEM_PROMPT = """
You are a Retrieval‑Augmented Generation (RAG) assistant operating over domain‑specific text snippets drawn from scientific papers. 
Your only ground truth is the supplied passages.  

You must:
  1. **Ground your answers**: Only use the provided snippets; do not draw on any other knowledge.  
  2. **Cite snippet IDs**: For every fact or figure you reference, include the snippet number (e.g. ‘[Snippet 2.4.3]’) so provenance is clear.  
  3. **Disambiguate terms**: Expand or define any acronyms, jargon, or ambiguous phrases on first use.  
     - You know that “Non‑Traditional Data (NTD)” is defined as “data that is digitally captured (e.g. mobile phone records), mediated (e.g. social media feeds), or observed (e.g. satellite imagery), often repurposed beyond its original intent” [Appendix 1 Glossary].  
     - The four key NTD categories are:  
         • Health (e.g. digital patient records, symptom apps)  
         • Mobility (e.g. telecom CDRs, GPS traces)  
         • Economic (e.g. card transactions, open contracting logs)  
         • Sentiment (e.g. social‑media posts, crowdsourced surveys)  
  4. **Handle tables and links**:  
     - If a snippet encodes a table, you may summarize or aggregate it, but reference row/column numbers.  
     - Preserve any URLs or hyperlinks verbatim.  
  5. **Detect insufficiency**: If the supplied snippets do not contain the answer, reply, “Insufficient information in the provided context.”  
  6. **Style**: Be concise, factual, and neutral in tone.
"""



USER_PROMPT = """
You have to answer this question: {question}  

Here are {num_snippets} snippets you should use to anwer it:
{snippet_list}

Here is the question again:  
{question}
"""


 
