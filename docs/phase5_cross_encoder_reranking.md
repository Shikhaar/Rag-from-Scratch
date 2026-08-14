# Phase 5: Cross-Encoder Neural Reranking

Phase 5 introduces a two-stage retrieval architecture: using fast candidate retrieval (Dense + BM25) to narrow down hundreds or thousands of documents to top 20–50 candidates, followed by an intensive neural **Cross-Encoder** to determine the final 3–5 chunks sent to the LLM.

---

## 1. Bi-Encoder vs. Cross-Encoder Architecture

### Bi-Encoder (Dense Embeddings Stage)
```
Query ──────> [ BERT / MiniLM ] ──────> Vector q (384-d)
                                               │
                                               ▼  Cosine Similarity (dot product)
                                               ▲
Document ───> [ BERT / MiniLM ] ──────> Vector d (384-d)
```
- **Pros**: Independent vector pre-computation, sub-millisecond similarity search.
- **Cons**: No cross-attention between individual query words and document words during encoding.

### Cross-Encoder (Reranker Stage)
```
[CLS] Query Tokens [SEP] Document Tokens [SEP]
                    │
                    ▼
          [ All-to-All Self-Attention ]
                    │
                    ▼
           Relevance Score (Logit)
```
- **Pros**: Full cross-attention allows query terms to directly interact with document terms in every Transformer layer, capturing rich negation, qualifiers, and nuanced context.
- **Cons**: $O(N)$ inference cost per query; computationally prohibitive for thousands of documents, but optimal for top 20–50 candidates.

---

## 2. The Two-Stage Funnel

```
Corpus (10,000+ chunks)
        │
        ▼  [ Stage 1: Dense + BM25 Retrieval ]
Top 20–50 Candidate Chunks
        │
        ▼  [ Stage 2: Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`) ]
Top 3–5 Final Chunks (Highest Precision)
        │
        ▼  [ Context Assembly ]
LLM Generation
```

Our `CrossEncoderReranker` maps predicted scores directly onto `SearchResult.rerank_score`, sorting candidates descending to ensure only the most authoritative contexts are presented to the LLM prompt.
