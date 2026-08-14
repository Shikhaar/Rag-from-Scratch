# Phase 4: Hybrid Retrieval & Fusion Algorithms

Dense semantic search and sparse BM25 keyword search exhibit complementary strengths and weaknesses:
- **Dense Vectors**: Excel at conceptual semantic queries, paraphrasing, and synonyms; struggle with exact numbers, product IDs, and rare jargon.
- **BM25 Sparse**: Excels at exact technical tokens, IDs, and domain jargon; fails when queries use alternate vocabulary or semantic abstractions.

Phase 4 fuses both retrieval paradigms into a single unified candidate ranking pipeline.

---

## 1. Reciprocal Rank Fusion (RRF)

RRF is a robust, parameter-insensitive rank fusion algorithm that evaluates the position of items across different rankings rather than raw scores (which have incompatible scales):

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### Mathematical Properties:
- $M = \{\text{dense}, \text{sparse}\}$: Set of retrievers.
- $r_m(d)$: 1-based rank of document $d$ in retriever $m$.
- $k = 60$: Standard smoothing constant (empirically derived in IR literature) that prevents high-ranking outliers from dominating the fused score.

### Example Walkthrough:
If Document A is Rank 1 in Dense and Rank 2 in BM25:
$$\text{RRF}(A) = \frac{1}{60 + 1} + \frac{1}{60 + 2} = \frac{1}{61} + \frac{1}{62} \approx 0.01639 + 0.01613 = 0.03252$$

If Document B is Rank 10 in Dense and absent from BM25:
$$\text{RRF}(B) = \frac{1}{60 + 10} = \frac{1}{70} \approx 0.01429$$

Document A is ranked higher due to reciprocal reinforcement across both retrieval channels.

---

## 2. Linear Score Fusion (Alpha Blending)

As an alternative to RRF, Min-Max normalization maps raw dense and sparse scores to $[0, 1]$, followed by a convex linear combination:

$$\text{Score}_{\text{norm}}(d) = \frac{\text{Score}(d) - \min(S)}{\max(S) - \min(S) + \epsilon}$$

$$\text{Score}_{\text{hybrid}}(d) = \alpha \cdot \text{Score}_{\text{dense, norm}}(d) + (1 - \alpha) \cdot \text{Score}_{\text{bm25, norm}}(d)$$

Where $\alpha \in [0, 1]$ allows tuning the preference balance between semantic understanding and exact keyword matching.
