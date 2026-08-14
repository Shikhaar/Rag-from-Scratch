"""
Unit tests for Okapi BM25 sparse keyword search index.
"""

import unittest
from core.schema import Chunk
from core.bm25 import BM25Index


class TestBM25(unittest.TestCase):

    def test_bm25_keyword_ranking(self):
        index = BM25Index(k1=1.5, b=0.75)

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="The Raft consensus algorithm handles leader election.", metadata={}),
            Chunk(chunk_id="c2", doc_id="d2", content="Deep neural networks use AdamW optimizer for weight decay.", metadata={}),
            Chunk(chunk_id="c3", doc_id="d3", content="Consensus in distributed systems requires majority quorum election.", metadata={}),
        ]

        index.add_chunks(chunks)

        results = index.search("Raft consensus leader election", top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertTrue(results[0].bm25_score > 0.0)

    def test_bm25_term_saturation(self):
        # A doc with 10 repeated keywords vs 1 keyword should have bounded score increase (due to k1)
        index = BM25Index(k1=1.5, b=0.75)
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="quantum quantum quantum quantum quantum computing", metadata={}),
            Chunk(chunk_id="c2", doc_id="d2", content="quantum physics fundamentals", metadata={}),
        ]
        index.add_chunks(chunks)

        results = index.search("quantum", top_k=2)
        score_c1 = results[0].score if results[0].chunk.chunk_id == "c1" else results[1].score
        score_c2 = results[1].score if results[0].chunk.chunk_id == "c1" else results[0].score

        # c1 has 5x term frequency, but score is dampened by (k1+1) saturation curve
        self.assertTrue(score_c1 > score_c2)
        self.assertTrue(score_c1 < score_c2 * 4.0)


if __name__ == "__main__":
    unittest.main()
