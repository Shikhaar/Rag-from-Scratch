"""
Evaluation metrics implemented from first principles.
Includes: Recall@K, Precision@K, Hit Rate@K, MRR, NDCG@K, Faithfulness, Citation Accuracy.
"""

import math
import re
from typing import Any, Dict, List, Set


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """
    Recall@K = (Number of relevant items in top-K) / (Total relevant items)
    """
    if not ground_truth_ids:
        return 1.0
    top_k_retrieved = retrieved_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in ground_truth_ids)
    return hits / len(ground_truth_ids)


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """
    Precision@K = (Number of relevant items in top-K) / K
    """
    if k == 0:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    hits = sum(1 for cid in top_k_retrieved if cid in ground_truth_ids)
    return hits / k


def hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant item appears in top-K, else 0.0
    """
    if not ground_truth_ids:
        return 1.0
    top_k_retrieved = retrieved_ids[:k]
    for cid in top_k_retrieved:
        if cid in ground_truth_ids:
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """
    MRR = 1 / rank of the first relevant document. Returns 0.0 if not found.
    """
    if not ground_truth_ids:
        return 1.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance:
    DCG@K = sum((2^rel - 1) / log2(rank + 1))
    IDCG@K = Ideal DCG for min(len(ground_truth), K) relevant items at the top.
    """
    if not ground_truth_ids:
        return 1.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, cid in enumerate(top_k, 1):
        rel = 1.0 if cid in ground_truth_ids else 0.0
        if rel > 0:
            dcg += (math.pow(2, rel) - 1.0) / math.log2(rank + 1)

    # Compute Ideal DCG
    ideal_hits = min(len(ground_truth_ids), k)
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += (math.pow(2, 1.0) - 1.0) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def citation_accuracy(generated_answer: str, num_retrieved_sources: int) -> Dict[str, Any]:
    """
    Measures validity and density of citations in the generated response.
    Checks that every citation tag [Source X] references a valid retrieved source index.
    """
    citations_found = re.findall(r"\[Source\s+(\d+)\]", generated_answer)
    if not citations_found:
        return {
            "has_citations": False,
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "citation_accuracy": 0.0,
        }

    valid = 0
    invalid = 0
    for c in citations_found:
        source_idx = int(c)
        if 1 <= source_idx <= num_retrieved_sources:
            valid += 1
        else:
            invalid += 1

    accuracy = valid / len(citations_found) if citations_found else 0.0
    return {
        "has_citations": True,
        "total_citations": len(citations_found),
        "valid_citations": valid,
        "invalid_citations": invalid,
        "citation_accuracy": accuracy,
    }


def keyword_overlap_faithfulness(generated_answer: str, context_text: str) -> float:
    """
    Heuristic lexical faithfulness score:
    Measures the ratio of key content terms in the generated answer that appear in the context.
    """
    if not generated_answer.strip() or not context_text.strip():
        return 0.0

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "of", "for", "with", "that", "this", "it"}
    answer_tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", generated_answer.lower()) if t not in stop_words]
    if not answer_tokens:
        return 1.0

    context_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", context_text.lower()))
    supported = sum(1 for t in answer_tokens if t in context_tokens)
    return supported / len(answer_tokens)
