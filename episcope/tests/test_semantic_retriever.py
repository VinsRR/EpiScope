import unittest
from unittest.mock import Mock
import tempfile
import shutil

from episcope.rag.retrieval.semantic import SemanticRetriever
from episcope.vectordb.faiss import FaissDB
from episcope.rag.embeddings import SimplifiedEmbedder
from episcope.rag.retrieval.hyde import HYDE

class TestSemanticRetriever(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db = FaissDB(index_dir=self.test_dir)
        self.embedder = SimplifiedEmbedder(embed_model="distilbert-base-uncased")
        
        docs = [
            {"content": "Epidemiological studies show patterns."},
            {"content": "Literature reviews summarise prior work."},
        ]
        points_to_upsert = []
        for i, doc in enumerate(docs):
            points_to_upsert.append({
                "id": f"doc{i}",
                "vector": self.embedder.embed_text(doc["content"]),
                "payload": {"text": doc["content"]}
            })
        self.db.upsert(points_to_upsert, namespace="test_ns")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_retrieve_without_hyde(self) -> None:
        retriever = SemanticRetriever(embedder=self.embedder, vectordb=self.db)
        contexts = retriever.retrieve("patterns", top_k=2, paper_id="test_ns")
        self.assertEqual(len(contexts), 2)
        self.assertTrue(hasattr(contexts[0], 'text'))
        self.assertEqual(contexts[0].text, "Epidemiological studies show patterns.")

    def test_retrieve_with_hyde(self) -> None:
        hyde = Mock(spec=HYDE)
        hyde.generate.return_value = "This is a hypothesis about patterns."
        
        retriever = SemanticRetriever(embedder=self.embedder, vectordb=self.db, hyde=hyde)
        contexts = retriever.retrieve("data", top_k=1, paper_id="test_ns")
        
        hyde.generate.assert_called_once()
        self.assertEqual(len(contexts), 1)
        self.assertTrue(hasattr(contexts[0], 'text'))

if __name__ == "__main__":
    unittest.main()
