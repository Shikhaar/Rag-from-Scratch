"""
Unit tests for evaluation metrics (Recall@K, Precision@K, HitRate, MRR, NDCG@K, Citations).
"""

import unittest
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    citation_accuracy,
)


class TestMetrics(unittest.TestCase):

    def test_recall_at_k(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        ground_truth = {"doc2", "doc4"}

        # At k=1: retrieved ["doc1"], hits = 0 -> 0.0
        self.assertEqual(recall_at_k(retrieved, ground_truth, k=1), 0.0)
        # At k=2: retrieved ["doc1", "doc2"], hits = 1 / 2 -> 0.5
        self.assertEqual(recall_at_k(retrieved, ground_truth, k=2), 0.5)
        # At k=4: hits = 2 / 2 -> 1.0
        self.assertEqual(recall_at_k(retrieved, ground_truth, k=4), 1.0)

    def test_mrr(self):
        # First relevant at rank 2 -> MRR = 1/2 = 0.5
        self.assertEqual(mean_reciprocal_rank(["doc1", "doc2", "doc3"], {"doc2"}), 0.5)
        # First relevant at rank 1 -> MRR = 1.0
        self.assertEqual(mean_reciprocal_rank(["doc1", "doc2"], {"doc1"}), 1.0)
        # Not found -> MRR = 0.0
        self.assertEqual(mean_reciprocal_rank(["doc1", "doc2"], {"doc3"}), 0.0)

    def test_ndcg_at_k(self):
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1"}
        # Ideal ranking: doc1 at rank 1 -> NDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k(retrieved, ground_truth, k=3), 1.0, places=4)

    def test_citation_accuracy(self):
        answer = "The policy allows refunds within 30 days [Source 1], verified in [Source 2]."
        metrics = citation_accuracy(answer, num_retrieved_sources=2)
        self.assertTrue(metrics["has_citations"])
        self.assertEqual(metrics["valid_citations"], 2)
        self.assertEqual(metrics["invalid_citations"], 0)
        self.assertEqual(metrics["citation_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
