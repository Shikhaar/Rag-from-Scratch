"""
Retrievers: Dense, Sparse (BM25), and Hybrid (Reciprocal Rank Fusion & Alpha Blending).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
from core.schema import Chunk, SearchResult
from core.vector_store import NumpyVectorStore
from core.bm25 import BM25Index
from core.embeddings import BaseEmbeddings

FusionMethod = Literal["rrf", "linear"]


class BaseRetriever(ABC):
    """Abstract interface for all retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieve top_k matching chunks for a query."""
        pass


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using embeddings and NumPy vector store."""

    def __init__(self, vector_store: NumpyVectorStore, embeddings: BaseEmbeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_vec = self.embeddings.embed_query(query)
        return self.vector_store.search(query_vec, top_k=top_k, filters=filters)


class SparseRetriever(BaseRetriever):
    """Sparse keyword retriever using custom Okapi BM25 inverted index."""

    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        return self.bm25_index.search(query, top_k=top_k, filters=filters)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever fusing dense vector similarity and sparse BM25 keyword matching.
    Supports Reciprocal Rank Fusion (RRF) and Linear Score Blending.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_method: FusionMethod = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,  # Weight for dense in linear blending (1-alpha for BM25)
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        candidate_multiplier: int = 3,
    ) -> List[SearchResult]:
        """
        Execute dense and sparse retrievals and fuse their candidate sets.
        """
        fetch_k = max(top_k * candidate_multiplier, 20)

        # 1. Fetch dense candidates
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        # 2. Fetch sparse candidates
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k, filters=filters)

        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]

        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        elif self.fusion_method == "linear":
            return self._linear_score_fusion(dense_results, sparse_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        RRF Algorithm:
        RRF_Score(d) = sum( 1 / (k + rank_m(d)) ) for m in [dense, sparse]
        """
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        dense_score_map: Dict[str, float] = {}
        sparse_score_map: Dict[str, float] = {}

        # Process dense ranks
        for rank, res in enumerate(dense_results, 1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            dense_score_map[cid] = res.score
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Process sparse ranks
        for rank, res in enumerate(sparse_results, 1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            sparse_score_map[cid] = res.score
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by RRF score
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[SearchResult] = []
        for rank, (cid, rrf_score) in enumerate(sorted_chunks, 1):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=float(rrf_score),
                    dense_score=dense_score_map.get(cid),
                    bm25_score=sparse_score_map.get(cid),
                    rrf_score=float(rrf_score),
                    rank=rank,
                )
            )

        return results

    def _linear_score_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Min-Max normalize scores and combine linearly: alpha * dense + (1 - alpha) * sparse.
        """
        chunk_map: Dict[str, Chunk] = {}
        dense_norm: Dict[str, float] = {}
        sparse_norm: Dict[str, float] = {}
        raw_dense: Dict[str, float] = {}
        raw_sparse: Dict[str, float] = {}

        # Normalize dense scores
        d_scores = [r.score for r in dense_results]
        min_d, max_d = min(d_scores), max(d_scores)
        denom_d = (max_d - min_d) if (max_d - min_d) > 1e-6 else 1.0
        for r in dense_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            raw_dense[cid] = r.score
            dense_norm[cid] = (r.score - min_d) / denom_d

        # Normalize sparse scores
        s_scores = [r.score for r in sparse_results]
        min_s, max_s = min(s_scores), max(s_scores)
        denom_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0
        for r in sparse_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            raw_sparse[cid] = r.score
            sparse_norm[cid] = (r.score - min_s) / denom_s

        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        fused_scores: Dict[str, float] = {}

        for cid in all_ids:
            d_val = dense_norm.get(cid, 0.0)
            s_val = sparse_norm.get(cid, 0.0)
            fused_scores[cid] = self.alpha * d_val + (1.0 - self.alpha) * s_val

        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[SearchResult] = []
        for rank, (cid, fused_score) in enumerate(sorted_chunks, 1):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=float(fused_score),
                    dense_score=raw_dense.get(cid),
                    bm25_score=raw_sparse.get(cid),
                    rank=rank,
                )
            )

        return results
