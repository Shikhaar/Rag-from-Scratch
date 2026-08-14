# Phase 2: NumPy Dense Vector Store

Phase 2 implements the dense geometric indexing layer entirely from scratch using **NumPy matrix operations**, eliminating black-box vector databases (FAISS, Chroma, Qdrant, Pinecone).

---

## 1. Mathematical Similarity Formulations

Given a query vector $\mathbf{q} \in \mathbb{R}^D$ and document matrix $\mathbf{V} \in \mathbb{R}^{N \times D}$:

### 1. Cosine Similarity (Default)
Measures the directional alignment regardless of vector magnitude:

$$\text{CosineSim}(\mathbf{q}, \mathbf{v}_i) = \frac{\mathbf{q} \cdot \mathbf{v}_i}{\|\mathbf{q}\|_2 \|\mathbf{v}_i\|_2} = \frac{\sum_{j=1}^D q_j v_{i,j}}{\sqrt{\sum_{j=1}^D q_j^2} \sqrt{\sum_{j=1}^D v_{i,j}^2}}$$

In vectorized NumPy:
```python
dot_products = np.dot(self.vectors, query_vec)
doc_norms = np.linalg.norm(self.vectors, axis=1)
query_norm = np.linalg.norm(query_vec)
scores = dot_products / (doc_norms * query_norm + 1e-10)
```

### 2. Dot Product
Used when embeddings are pre-normalized to unit sphere ($\|\mathbf{v}\|_2 = 1$):

$$\text{DotProduct}(\mathbf{q}, \mathbf{v}_i) = \mathbf{V} \mathbf{q}^T$$

### 3. Euclidean Distance & Similarity Mapping
Computes geometric $L_2$ distance, inverted into a $[0, 1]$ similarity score:

$$d(\mathbf{q}, \mathbf{v}_i) = \|\mathbf{q} - \mathbf{v}_i\|_2 = \sqrt{\sum_{j=1}^D (q_j - v_{i,j})^2}$$

$$\text{Sim}_{\text{euclidean}} = \frac{1}{1 + d(\mathbf{q}, \mathbf{v}_i)}$$

---

## 2. Fast Top-$K$ Selection

For small candidate sizes ($N \le K$), standard `np.argsort(-scores)` is used. For large corpora ($N > K$), efficient $O(N)$ partitioning is applied:

```python
partition_idx = np.argpartition(-valid_scores, top_k)[:top_k]
sorted_order = partition_idx[np.argsort(-valid_scores[partition_idx])]
```

---

## 3. Metadata Filtering

Filters allow constraint search (e.g. `filters={"source": "cloud_policy.txt", "page": 1}`):
- Evaluated prior to top-$K$ selection.
- Allows combined semantic similarity with strict organizational/security partition boundaries.

---

## 4. Atomic Disk Serialization

Persistence stores both dense arrays and metadata cleanly:
- **`vectors.npz`**: Compressed NumPy binary representation of the $(N, D)$ embedding matrix.
- **`chunks_metadata.json`**: Structured JSON catalog containing chunk text, source metadata, character offsets, and chunk hashes.
