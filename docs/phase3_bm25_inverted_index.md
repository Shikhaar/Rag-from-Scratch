# Phase 3: Okapi BM25 Sparse Inverted Index

Phase 3 implements the Okapi BM25 sparse keyword search algorithm from mathematical first principles. While dense embeddings capture high-level semantic intent, BM25 excels at exact keyword matching, acronyms, technical identifiers, and precise alphanumeric search terms.

---

## 1. Mathematical Formulation

The Okapi BM25 score of a document $D$ for query $Q = \{q_1, q_2, \dots, q_n\}$ is defined as:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### Variables:
- $f(q_i, D)$: Term frequency of query term $q_i$ in document $D$.
- $|D|$: Length of document $D$ in tokens.
- $\text{avgdl}$: Average token length of all documents in the corpus.
- $k_1 = 1.5$: Term frequency saturation parameter. Controls how quickly repeated occurrences of a word reach diminishing returns.
- $b = 0.75$: Document length normalization penalty. Penalizes long documents that contain terms merely due to length.

---

## 2. Robertson-Spärck Jones IDF (Non-Negative)

Traditional IDF formulations can yield negative values for high-frequency terms appearing in $> 50\%$ of documents. Our implementation uses the smoothed Robertson-Spärck Jones non-negative IDF:

$$\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$

Where:
- $N$: Total number of document chunks in the index.
- $n(q_i)$: Number of document chunks containing term $q_i$.

---

## 3. Inverted Index Data Structures

```python
class BM25Index(BaseInvertedIndex):
    # Inverted index: term -> {chunk_index: term_frequency}
    self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
    
    # Document frequencies: term -> count of docs containing term
    self.doc_frequencies: Dict[str, int] = defaultdict(int)
    
    # Precomputed Robertson-Spärck Jones IDFs
    self.idf_cache: Dict[str, float] = {}
```

### Search Process
1. Query string is tokenized and filtered against English stopwords.
2. For each query term, its posting list in `inverted_index` is traversed.
3. Length-normalized TF components are accumulated for each matching document.
4. Candidates are sorted descending by BM25 score and returned as `SearchResult` objects.
