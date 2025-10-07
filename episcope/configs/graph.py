# configs/graph_rag_config.py

# Neo4j
# NEO4J_URL  = "bolt://neo4j:7687" # 
NEO4J_URL  = "bolt://localhost:7687" #"7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"
DEFAULT_DB = "mobilitywp2" # NO UNDERSCORES OR SPECIAL CHARACTERS!

# LLM & embeddings
LLAMA_MODEL = "llama3.2-vision" # "llama3.2-vision" # llama3.2:1b
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TIMEOUT     = 300
