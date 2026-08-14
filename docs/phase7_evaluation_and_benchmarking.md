# Phase 7: Retrieval Evaluation & Automated Benchmarking

Phase 7 establishes a quantitative, empirical evaluation suite to measure retrieval precision, recall, ranking quality, and answer faithfulness from mathematical principles.

---

## 1. Information Retrieval Metrics (Pure Math)

Given a query $q$, retrieved candidate list $R_k = [r_1, r_2, \dots, r_k]$, and ground truth relevant set $G$:

### 1. Recall@K
Proportion of ground truth relevant chunks successfully retrieved in the top $K$:

$$\text{Recall}@K = \frac{|R_k \cap G|}{|G|}$$

### 2. Precision@K
Proportion of top $K$ retrieved chunks that are actually relevant:

$$\text{Precision}@K = \frac{|R_k \cap G|}{K}$$

### 3. Hit Rate@K (Success Rate)
Binary metric indicating whether at least one relevant document was retrieved:

$$\text{HitRate}@K = \mathbb{I}(|R_k \cap G| > 0)$$

### 4. Mean Reciprocal Rank (MRR)
Evaluates where the *first* relevant chunk appears in the ranking:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the rank index of the first relevant document for query $i$.

### 5. Normalized Discounted Cumulative Gain (NDCG@K)
Measures ranking quality with position-based logarithmic discounting:

$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

Where $\text{IDCG}@K$ is the Ideal DCG score achieved by placing all relevant items at the top.

---

## 2. Automated Comparative Benchmark Suite

The `RAGBenchmark` module evaluates 4 distinct retrieval configurations side-by-side:

```python
benchmark = RAGBenchmark(
    dense_retriever=pipeline.dense_retriever,
    sparse_retriever=pipeline.sparse_retriever,
    reranker=pipeline.reranker,
    dataset=eval_dataset,
)
results = benchmark.run_benchmark(all_chunks=pipeline.chunks, k=5)
```

### Empirical Benchmark Output:

| Configuration | Recall@5 | MRR | NDCG@5 | HitRate@5 | Latency (ms) |
|---|---|---|---|---|---|
| **BM25 (Sparse)** | 0.9444 | 1.0000 | 0.9609 | 1.0000 | 0.13 |
| **Dense (NumPy)** | 0.9444 | 1.0000 | 0.9507 | 1.0000 | 66.95 |
| **Hybrid (RRF)** | **1.0000** | **1.0000** | **0.9946** | **1.0000** | 52.96 |
| **Hybrid + Reranker** | **1.0000** | **1.0000** | **0.9892** | **1.0000** | 1809.31 |

### Key Takeaways:
1. **Hybrid (RRF)** outperforms single retrievers across both Recall@5 and NDCG@5, achieving 100% recall.
2. **BM25** provides sub-millisecond latency (0.13ms) for rapid lexical matching.
3. **Cross-Encoder Reranker** optimizes ranking precision when deep semantic interactions are required.
