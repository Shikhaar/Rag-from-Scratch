"""
Evaluator for measuring retrieval performance metrics across query datasets.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set
from core.schema import Chunk, SearchResult
from core.retriever import BaseRetriever
from core.reranker import BaseReranker
from evaluation.dataset import EvalDataset, EvalSample
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)


class RetrievalEvaluator:
    """
    Evaluates a retriever or (retriever + reranker) pipeline against a labeled dataset.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 5,
        rerank_candidate_k: int = 20,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_candidate_k = rerank_candidate_k

    def _resolve_ground_truth_chunks(
        self, sample: EvalSample, all_chunks: List[Chunk]
    ) -> Set[str]:
        """
        Identify matching chunk IDs based on explicit IDs, filenames, or keywords.
        """
        ground_truth_ids = set(sample.relevant_chunk_ids)

        if sample.relevant_doc_names:
            for chunk in all_chunks:
                fname = chunk.metadata.get("filename", "")
                src = chunk.metadata.get("source", "")
                for doc_name in sample.relevant_doc_names:
                    if doc_name.lower() in fname.lower() or doc_name.lower() in src.lower():
                        # If keywords specified, check if chunk contains at least one
                        if sample.relevant_keywords:
                            content_lower = chunk.content.lower()
                            if any(kw.lower() in content_lower for kw in sample.relevant_keywords):
                                ground_truth_ids.add(chunk.chunk_id)
                        else:
                            ground_truth_ids.add(chunk.chunk_id)

        return ground_truth_ids

    def evaluate_sample(
        self,
        sample: EvalSample,
        all_chunks: List[Chunk],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """Evaluate a single query sample."""
        gt_ids = self._resolve_ground_truth_chunks(sample, all_chunks)
        if not gt_ids:
            # If no ground truth chunks exist in the corpus, skip or return empty
            return {}

        start_t = time.perf_counter()

        # Step 1: Initial Retrieval
        fetch_k = self.rerank_candidate_k if self.reranker else max(k_values)
        results = self.retriever.retrieve(sample.query, top_k=fetch_k)

        # Step 2: Optional Reranker
        if self.reranker and results:
            results = self.reranker.rerank(sample.query, results, top_k=max(k_values))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        retrieved_chunk_ids = [r.chunk.chunk_id for r in results]

        metrics = {
            "query": sample.query,
            "latency_ms": elapsed_ms,
            "mrr": mean_reciprocal_rank(retrieved_chunk_ids, gt_ids),
        }

        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_chunk_ids, gt_ids, k=k)
            metrics[f"precision@{k}"] = precision_at_k(retrieved_chunk_ids, gt_ids, k=k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_chunk_ids, gt_ids, k=k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_chunk_ids, gt_ids, k=k)

        return metrics

    def evaluate(
        self,
        dataset: EvalDataset,
        all_chunks: List[Chunk],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Evaluate full dataset and compute aggregate mean metrics.
        """
        sample_metrics = []
        for sample in dataset.samples:
            res = self.evaluate_sample(sample, all_chunks, k_values=k_values)
            if res:
                sample_metrics.append(res)

        if not sample_metrics:
            return {"error": "No valid samples with matching ground truth chunks."}

        # Aggregate averages
        num_samples = len(sample_metrics)
        agg: Dict[str, float] = {
            "num_samples": num_samples,
            "avg_latency_ms": sum(m["latency_ms"] for m in sample_metrics) / num_samples,
            "MRR": sum(m["mrr"] for m in sample_metrics) / num_samples,
        }

        for k in k_values:
            agg[f"Recall@{k}"] = sum(m[f"recall@{k}"] for m in sample_metrics) / num_samples
            agg[f"Precision@{k}"] = sum(m[f"precision@{k}"] for m in sample_metrics) / num_samples
            agg[f"HitRate@{k}"] = sum(m[f"hit_rate@{k}"] for m in sample_metrics) / num_samples
            agg[f"NDCG@{k}"] = sum(m[f"ndcg@{k}"] for m in sample_metrics) / num_samples

        return {
            "summary": agg,
            "sample_details": sample_metrics,
        }
