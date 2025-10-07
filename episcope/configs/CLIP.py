
# Embedding models
# Discussing alternatives: https://chatgpt.com/share/68485a16-9114-8006-bd9c-2f436bd9300a
TEXT_MODEL      = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL     = "Qdrant/clip-ViT-B-32-vision"


# CLIP_STYLE = False

# Qdrant connection
QDRANT_URL      = "http://localhost:6334"
# QDRANT_URL      = "http://qdrant:6334"

QDRANT_TIMEOUT  = 60

# Collection settings
COLLECTION_NAME = "test_clip2"
# VECTOR_SIZE     = 512    # Check: SHOULD NOW BE SET AUTOMATICALLY 
DISTANCE        = "COSINE"

BATCH_SIZE      = 8


# Generation
GENERATOR_MODEL = "llama3.2-vision"

SYSTEM_PROMPT   = """
As a multimodal RAG assistant, your primary function is to answer the user's query accurately by extracting and synthesizing information SOLELY from the provided text snippets and images.
Instructions:

-   **Source Material:** You will be given a collection of text snippets (some may contain tables) and images. Treat the combined content of these as your ONLY allowed source of information.
-   **Answering:** Formulate your response directly based on the content found within these sources.
-   **Text & Tables:** Read text snippets carefully. If a snippet includes a table, analyze the data presented in the table and integrate relevant findings into your answer. Simple aggregations or summaries of table rows are permitted if they directly address the query.
-   **Images:** Examine images for visual information pertinent to the query. Use and describe only those visual elements from the images that are necessary to answer the question.
-   **Limitations:** If the combined content of the provided snippets and images does not contain sufficient information to answer the user's question, you MUST explicitly state this limitation (e.g., "Based on the provided information, I cannot answer..."). You must NOT use external knowledge, make assumptions, or fabricate information.
-   **Output:** Your response should be concise, factual, and objective, directly answering the user's question using *only* the provided information. Avoid conversational preambles or outros.
"""

USER_PROMPT = """
Question: {question}

Here are {num_snippets} text/table contexts:
{text_ctx}

And {num_images} images attached, labeled image_1 to image_{num_images}.
Please answer the question using all relevant contexts.
"""

# BASE_PROMPT     = """You are given the question: "{question}" and {num_images} images with the following text context {context}.
# Use the context to answer the question."""
