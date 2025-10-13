# ColPali
COLPALI_MODEL =   "Qdrant/colpali-v1.3-fp16"   #"vidore/colpali-v1.2" # "vidore/colpali-v1.3-hf"  #
DEVICE        = "auto"      # or "cuda"/"mps"/"cpu"
BATCH_SIZE    = 4

# Qdrant
QDRANT_URL     = "http://localhost:6334" #"http://localhost:6333"
QDRANT_TIMEOUT = 60

# Collection
COLLECTION_NAME = "test_colpali" #"colpali_collection"
VECTOR_SIZE     = 128
DISTANCE        = "COSINE"


GENERATOR_MODEL = "llama3.2-vision"
# BASE_PROMPT = """You are given the question: "{question}" and {num_images} images."""



SYSTEM_PROMPT = """
You are an AI assistant tasked with answering user questions by analyzing provided document page images.

Here are your instructions:
1.  Your ONLY source of information is the set of provided document page images. Treat the content within these images as the complete and sole basis for your response.
2.  Read and interpret the visual content of the provided pages carefully.
3.  Answer the user's question using *only* the facts and information that you find directly on these provided pages. Do not incorporate any outside knowledge or make assumptions.
4.  If a page contains visible tables, charts, diagrams, or embedded text, analyze and interpret this visual content as needed to form your answer, but base your interpretation only on what is directly shown on the page.
5.  If the provided pages DO NOT contain sufficient information to fully answer the user’s question, you MUST explicitly state that the information is not available in the provided documents. Do not guess, speculate, or invent information.
6.  Present your answer concisely, factually, and objectively.
"""

USER_PROMPT = """
Here are {num_pages} pages of document images:

Question: {question}

Please answer the question using *only* the visual content of these pages. 
"""





SYNTHESIS_PROMPT = """
You have received the following individual analyses for the same question:

{combined_text}

Now, synthesize those into a single, coherent answer to:

"{question}"

Use only the *relevant* insights from each analysis.  
Be concise, factual, and ensure your final answer is fully grounded in the provided contexts.
"""

SYNTHESIS_MODEL = "llama3.2:1b"  # "llama3.2:1b"  # "llama3.2-vision"