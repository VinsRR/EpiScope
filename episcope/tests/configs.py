test_model_hf_embedding = "prajjwal1/bert-tiny"
test_model_ollama = "tinyllama:1.1b"



# import numpy as np
# class DummyEmbedder:
#     def embed_query(self, text):
#         return np.zeros(384).tolist()
#     def embed_documents(self, texts):
#         return [np.zeros(384).tolist() for _ in texts]