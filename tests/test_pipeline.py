"""
Unit tests for end-to-end RAGPipeline.
"""

import os
import shutil
import tempfile
import unittest

from rag_pipeline import RAGPipeline
from core.embeddings import HashingEmbeddings
from core.llm import MockLLM
from core.reranker import PassthroughReranker


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings = HashingEmbeddings(dimension=64)
        self.llm = MockLLM()
        self.reranker = PassthroughReranker()
        self.pipeline = RAGPipeline(
            embeddings=self.embeddings,
            llm=self.llm,
            reranker=self.reranker,
            index_dir=self.temp_dir,
            use_neural_models=False,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ingest_and_query_flow(self):
        texts = [
            "Raft is a consensus protocol that coordinates leader election and log replication.",
            "AdamW decouples weight decay from adaptive gradient momentum.",
            "Standard cloud refund policy offers 30-day full refund window.",
        ]
        metadatas = [
            {"source": "raft.md", "topic": "consensus"},
            {"source": "adamw.md", "topic": "ml"},
            {"source": "policy.txt", "topic": "billing"},
        ]

        stats = self.pipeline.ingest_texts(texts, metadatas=metadatas)
        self.assertEqual(stats["indexed_documents"], 3)
        self.assertEqual(stats["total_documents"], 3)

        # Test query
        res = self.pipeline.query("What is the refund policy window?")
        self.assertIsNotNone(res.answer)
        self.assertTrue(len(res.retrieved_chunks) > 0)
        self.assertEqual(res.retrieved_chunks[0].metadata.get("source"), "policy.txt")
        self.assertTrue(len(res.citations) > 0)

        # Test incremental indexing (re-ingesting same text skips)
        stats2 = self.pipeline.ingest_texts(texts, metadatas=metadatas)
        self.assertEqual(stats2["skipped_documents"], 3)
        self.assertEqual(stats2["indexed_documents"], 0)


if __name__ == "__main__":
    unittest.main()
