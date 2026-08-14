"""
Evaluation package for RAG From First Principles.
"""

from evaluation.dataset import EvalSample, EvalDataset
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    citation_accuracy,
    keyword_overlap_faithfulness,
)
from evaluation.evaluator import RetrievalEvaluator
from evaluation.benchmark import RAGBenchmark, print_benchmark_report

__all__ = [
    "EvalSample",
    "EvalDataset",
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "citation_accuracy",
    "keyword_overlap_faithfulness",
    "RetrievalEvaluator",
    "RAGBenchmark",
    "print_benchmark_report",
]
