"""
Unit tests for Dense, Sparse, and Hybrid (RRF) Retrievers.
"""

import unittest
import numpy as np
from core.schema import Chunk
from core.embeddings import HashingEmbeddings
from core.vector_store import NumpyVectorStore
from core.bm25 import BM25Index
from core.retriever import DenseRetriever, SparseRetriever, HybridRetriever


class TestRetrievers(unittest.TestCase):

    def setUp(self):
        self.embeddings = HashingEmbeddings(dimension=64)
        self.vector_store = NumpyVectorStore()
        self.bm25_index = BM25Index()

        self.chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Enterprise refund policy offers 30-day money back guarantee.", metadata={"source": "policy.txt"}),
            Chunk(chunk_id="c2", doc_id="d2", content="Consensus algorithms in distributed systems: Raft and Paxos.", metadata={"source": "raft.md"}),
            Chunk(chunk_id="c3", doc_id="d3", content="Neural network optimization with AdamW and FlashAttention.", metadata={"source": "ai.md"}),
        ]

        vecs = self.embeddings.embed_documents([c.content for c in self.chunks])
        self.vector_store.add_chunks(self.chunks, vecs)
        self.bm25_index.add_chunks(self.chunks)

        self.dense_retriever = DenseRetriever(self.vector_store, self.embeddings)
        self.sparse_retriever = SparseRetriever(self.bm25_index)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.sparse_retriever, fusion_method="rrf")

    def test_hybrid_rrf_retrieval(self):
        results = self.hybrid_retriever.retrieve("What is the refund policy?", top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertIsNotNone(results[0].rrf_score)
        self.assertTrue(results[0].rrf_score > 0.0)


if __name__ == "__main__":
    unittest.main()
