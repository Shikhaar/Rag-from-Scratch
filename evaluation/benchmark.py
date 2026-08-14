"""
Benchmark suite comparing BM25, Dense, Hybrid (RRF), and Hybrid + Reranker.
"""

from typing import Dict, List, Optional
import time
from tabulate import tabulate

from core.schema import Chunk
from core.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from core.reranker import BaseReranker, CrossEncoderReranker, PassthroughReranker
from evaluation.dataset import EvalDataset
from evaluation.evaluator import RetrievalEvaluator


class RAGBenchmark:
    """
    Automated comparative evaluation benchmark for RAG retrieval configurations.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: Optional[BaseReranker] = None,
        dataset: Optional[EvalDataset] = None,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion_method="rrf",
        )
        self.reranker = reranker or PassthroughReranker()
        self.dataset = dataset

    def run_benchmark(
        self,
        all_chunks: List[Chunk],
        dataset: Optional[EvalDataset] = None,
        k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run side-by-side evaluation across all retrieval configurations.
        """
        ds = dataset or self.dataset
        if not ds:
            raise ValueError("An EvalDataset must be provided to run the benchmark.")

        configs = [
            ("BM25 (Sparse)", self.sparse_retriever, None),
            ("Dense (NumPy)", self.dense_retriever, None),
            ("Hybrid (RRF)", self.hybrid_retriever, None),
            ("Hybrid + Reranker", self.hybrid_retriever, self.reranker),
        ]

        benchmark_results = {}

        for name, ret, rerank in configs:
            evaluator = RetrievalEvaluator(
                retriever=ret,
                reranker=rerank,
                top_k=k,
                rerank_candidate_k=20,
            )
            report = evaluator.evaluate(ds, all_chunks, k_values=[1, 3, 5, 10])
            if "summary" in report:
                benchmark_results[name] = report["summary"]
            else:
                benchmark_results[name] = {"error": 0.0}

        return benchmark_results

    def format_table(self, results: Dict[str, Dict[str, float]], k: int = 5) -> str:
        """
        Format benchmark results into a clean markdown / ASCII comparative table.
        """
        headers = [
            "Configuration",
            f"Recall@{k}",
            "MRR",
            f"NDCG@{k}",
            f"HitRate@{k}",
            "Latency (ms)",
        ]

        rows = []
        for config_name, metrics in results.items():
            if "error" in metrics:
                rows.append([config_name, "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue

            r_k = f"{metrics.get(f'Recall@{k}', 0.0):.4f}"
            mrr = f"{metrics.get('MRR', 0.0):.4f}"
            ndcg = f"{metrics.get(f'NDCG@{k}', 0.0):.4f}"
            hr = f"{metrics.get(f'HitRate@{k}', 0.0):.4f}"
            lat = f"{metrics.get('avg_latency_ms', 0.0):.2f}"

            rows.append([config_name, r_k, mrr, ndcg, hr, lat])

        table_str = tabulate(rows, headers=headers, tablefmt="github")
        return table_str


def print_benchmark_report(results: Dict[str, Dict[str, float]], k: int = 5) -> None:
    """Print a formatted benchmark comparison report."""
    benchmark = RAGBenchmark(None, None)  # type: ignore
    table = benchmark.format_table(results, k=k)
    print("\n" + "=" * 65)
    print("           RAG RETRIEVAL BENCHMARK COMPARISON REPORT")
    print("=" * 65)
    print(table)
    print("=" * 65 + "\n")
