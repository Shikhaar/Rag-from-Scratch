"""
Rerankers: Cross-Encoder reranking for precision refinement of top retrieval candidates.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from core.schema import SearchResult


class BaseReranker(ABC):
    """Abstract base class for candidate rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Rerank a list of SearchResults against query and return top_k."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Neural Cross-Encoder reranker using HuggingFace sentence-transformers.
    Evaluates (Query, Document) joint attention directly, capturing rich lexical & semantic interactions.
    Default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. Run `pip install sentence-transformers`."
            )

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not results:
            return []

        # Prepare pairs for cross-encoder scoring: [(query, chunk_text_1), ...]
        pairs = [(query, res.chunk.content) for res in results]
        raw_scores = self.model.predict(pairs)

        # Apply sigmoid normalization if logits
        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.ndim == 0:
            scores = np.array([scores.item()])

        # Map scores to results
        for res, score in zip(results, scores):
            res.rerank_score = float(score)

        # Sort descending by cross-encoder score
        reranked = sorted(results, key=lambda x: (x.rerank_score if x.rerank_score is not None else -float("inf")), reverse=True)
        top_results = reranked[:top_k]

        for rank, res in enumerate(top_results, 1):
            res.rank = rank

        return top_results


class PassthroughReranker(BaseReranker):
    """Fallback no-op reranker for testing or lightweight pipelines."""

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        for rank, res in enumerate(results[:top_k], 1):
            res.rank = rank
            res.rerank_score = res.score
        return results[:top_k]
