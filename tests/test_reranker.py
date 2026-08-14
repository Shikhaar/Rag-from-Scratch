"""
Unit tests for candidate rerankers (CrossEncoder & Passthrough).
"""

import unittest
from core.schema import Chunk, SearchResult
from core.reranker import PassthroughReranker


class TestReranker(unittest.TestCase):

    def test_passthrough_reranker(self):
        reranker = PassthroughReranker()
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Consensus in distributed systems."),
            Chunk(chunk_id="c2", doc_id="d2", content="Gradient descent optimization."),
        ]
        results = [
            SearchResult(chunk=chunks[0], score=0.9),
            SearchResult(chunk=chunks[1], score=0.8),
        ]

        reranked = reranker.rerank("distributed consensus", results, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].chunk.chunk_id, "c1")
        self.assertEqual(reranked[0].rank, 1)
        self.assertEqual(reranked[0].rerank_score, 0.9)


if __name__ == "__main__":
    unittest.main()
