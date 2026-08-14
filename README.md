# RAG From First Principles

> A production-oriented Retrieval-Augmented Generation system implemented from scratch in Python — including dense vector retrieval, Okapi BM25, hybrid search, cross-encoder reranking, citation grounding, incremental indexing, explainable inspection, and retrieval evaluation without LangChain, LlamaIndex, or monolithic frameworks.

---

## Architectural Blueprint

```
                         ┌─────────────────────────┐
                         │   Source Documents      │
                         │   (.txt, .md, .pdf)     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Ingestion & Hashing   │
                         │  (Incremental Indexing) │
                         └────────────┬────────────┘
                                      │
                               Chunk + Metadata
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
             Dense Embeddings                 BM25 Tokenizer
         (SentenceTransformers/API)         (Term & Doc Frequency)
                      │                               │
                      ▼                               ▼
             NumPy Vector Store              BM25 Inverted Index
           (Cosine/Dot/Euclidean)             (TF-IDF / Length Norm)
                      │                               │
                      └───────────────┬───────────────┘
                                      │
                              [ User Query ]
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
             Dense Similarity Search           BM25 Keyword Search
             (with Metadata Filters)         (with Metadata Filters)
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
                              Hybrid Retrieval
                            (RRF / α-Score Fusion)
                                      ▼
                            Top-N Candidates (20-50)
                                      ▼
                             Cross-Encoder Reranker
                                      ▼
                             Top-K Reranked (3-5)
                                      ▼
                           Prompt & Citation Assembly
                                      ▼
                             LLM Generation Engine
                           (Gemini / Ollama / Mock)
                                      ▼
                           Grounded Answer + Citations
                                      │
                                      ▼
                        Evaluation & Benchmark Suite
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
             Recall@K                MRR                NDCG@K
          (Hit Rate @ K)     (Mean Recip. Rank)   (Discounted Gain)
```

---

## Mathematical Foundations Implemented from Scratch

### 1. Dense Vector Similarity (NumPy Vector Store)
Cosine similarity measures the cosine of the angle between query vector $\mathbf{q}$ and document chunk vector $\mathbf{v}$:

$$\text{CosineSim}(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^D q_i v_i}{\sqrt{\sum_{i=1}^D q_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$

### 2. Okapi BM25 Sparse Inverted Index
The Robertson-Spärck Jones non-negative IDF and length-normalized term frequency formulation:

$$\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- $k_1 = 1.5$: Term frequency saturation parameter.
- $b = 0.75$: Document length penalization factor.
- $|D|$: Document length in tokens, $\text{avgdl}$: Average document length across corpus.

### 3. Reciprocal Rank Fusion (RRF)
RRF combines candidate rankings from disparate scoring distributions without requiring score calibration:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$

Where $r_m(d)$ is the 1-based rank of document $d$ in retriever $m$, and $k = 60$ is the smoothing constant.

### 4. Cross-Encoder Reranking
While dense bi-encoders encode query and document independently ($E(q) \cdot E(d)$), a neural Cross-Encoder computes joint all-to-all cross-attention:

$$\text{Score}_{\text{cross}}(q, d) = \text{CrossEncoder}(\mathbf{q} \circ \mathbf{d})$$

### 5. Evaluation Metrics
- **Recall@K**: $\frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$
- **Precision@K**: $\frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$
- **Mean Reciprocal Rank (MRR)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$
- **Normalized Discounted Cumulative Gain (NDCG@K)**: $\frac{\text{DCG}@K}{\text{IDCG}@K}$, where $\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$

---

## Repository Structure

```
rag-from-first-principles/
├── core/
│   ├── schema.py          # Document, Chunk, SearchResult, QueryResult dataclasses
│   ├── loaders.py         # Text, Markdown, PDF, Directory loaders with metadata
│   ├── chunker.py         # Character & Recursive boundary splitters
│   ├── embeddings.py      # Pretrained embeddings wrapper (SentenceTransformers, Gemini, Hashing fallback)
│   ├── vector_store.py    # NumPy Dense Vector Store (Cosine, Dot, Euclidean, Filters, Persistence)
│   ├── bm25.py            # Okapi BM25 Inverted Index from scratch (TF, IDF, doc-len norm)
│   ├── retriever.py       # Dense, Sparse, and Hybrid (RRF & Linear) retrievers
│   ├── reranker.py        # Cross-Encoder & Score-based Rerankers
│   ├── prompt.py          # Grounding templates & citation formatting
│   └── llm.py             # Gemini, Ollama, and Mock LLM providers
│
├── storage/
│   ├── index.py           # Unified index manager & incremental document hashing
│   └── persistence.py     # Clean serialization to disk (npz + json metadata)
│
├── evaluation/
│   ├── dataset.py         # QA & Ground Truth evaluation dataset structures
│   ├── metrics.py         # Pure math: Recall@K, Precision@K, MRR, NDCG@K, HitRate, Citations
│   ├── evaluator.py       # Evaluator for Dense, BM25, Hybrid, and Hybrid+Reranker
│   └── benchmark.py       # Automated benchmark runner with rich comparison table
│
├── data/
│   ├── sample_knowledge/  # Curated test documents (Distributed systems, Cloud policy, AI optimization)
│   └── eval_dataset.json  # Benchmark ground truth dataset
│
├── tests/                 # Comprehensive unit test suite (15 passing tests)
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   ├── test_bm25.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
│
├── rag_pipeline.py        # End-to-end RAG orchestrator
├── main.py                # Interactive CLI with /query, /inspect, /benchmark, /ingest, /stats
├── requirements.txt       # Clean dependencies
└── README.md              # Documentation
```

---

## Quickstart

### 1. Installation with Poetry (or pip)

```bash
# Using Poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. Interactive CLI

Launch the interactive REPL:

```bash
# Using Poetry script entrypoint
poetry run rag

# Or standard python
python main.py
```

### 3. Key CLI Commands

| Command | Description |
|---|---|
| `/query <question>` | Run grounded generation with CO-STAR prompt and inline citations |
| `/inspect <question>` | Inspect multi-stage scores (`Dense`, `BM25`, `RRF`, `Rerank`) |
| `/benchmark` | Run 4-way evaluation benchmark across the test dataset |
| `/ingest <path>` | Incrementally ingest a file (`.txt`, `.md`, `.pdf`) or directory |
| `/mode <hybrid\|dense\|bm25>` | Switch active retrieval strategy |
| `/stats` | View active index statistics and memory footprint |

---

## Complete Phase Documentation (`docs/`)

Explore in-depth mathematical derivations and design explanations:

- [**System Architecture & SOLID OOP Design**](file:///c:/Shikhar/Rag/docs/overview.md)
- [**Phase 1: Ingestion, Loaders & Recursive Chunking**](file:///c:/Shikhar/Rag/docs/phase1_ingestion_and_chunking.md)
- [**Phase 2: NumPy Dense Vector Store (Cosine, Euclidean, Dot Product)**](file:///c:/Shikhar/Rag/docs/phase2_dense_vector_store.md)
- [**Phase 3: Okapi BM25 Sparse Inverted Index from Scratch**](file:///c:/Shikhar/Rag/docs/phase3_bm25_inverted_index.md)
- [**Phase 4: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)**](file:///c:/Shikhar/Rag/docs/phase4_hybrid_retrieval_and_fusion.md)
- [**Phase 5: Cross-Encoder Neural Reranking**](file:///c:/Shikhar/Rag/docs/phase5_cross_encoder_reranking.md)
- [**Phase 6: CO-STAR Prompting Framework & Citation Grounding**](file:///c:/Shikhar/Rag/docs/phase6_costar_prompt_and_generation.md)
- [**Phase 7: Retrieval Evaluation & Automated Benchmarking**](file:///c:/Shikhar/Rag/docs/phase7_evaluation_and_benchmarking.md)

---

## Explainable Multi-Stage Inspection (`/inspect`)

Inspect granular scores across each step of the retrieval and ranking funnel:

```
rag [hybrid]> /inspect What is the refund period and policy?

Query: "What is the refund period and policy?"
──────────────────────────────────────────────────────────────────────
1. cloud_platform_policy.txt  (page: N/A, chunk: 0, id: 9a2f1b8c_0)
   Dense Score:  0.8124     BM25 Score:  7.8421
   RRF Score:    0.03279    Rerank Score: 0.9412
   "Enterprise customers are eligible for a full refund within a 30-day trial period..."

2. cloud_platform_policy.txt  (page: N/A, chunk: 1, id: 9a2f1b8c_1)
   Dense Score:  0.6432     BM25 Score:  3.1209
   RRF Score:    0.02105    Rerank Score: 0.4120
   "Requests submitted after 30 days but before 60 days are eligible for credits..."
```

---

## Automated Retrieval Benchmark

Run `python main.py benchmark` to evaluate all retrieval configurations on the dataset:

| Configuration | Recall@5 | MRR | NDCG@5 | HitRate@5 | Latency (ms) |
|---|---|---|---|---|---|
| **BM25 (Sparse)** | 0.9444 | 1.0000 | 0.9609 | 1.0000 | 0.13 |
| **Dense (NumPy)** | 0.9444 | 1.0000 | 0.9507 | 1.0000 | 66.95 |
| **Hybrid (RRF)** | **1.0000** | **1.0000** | **0.9946** | **1.0000** | 52.96 |
| **Hybrid + Reranker** | **1.0000** | **1.0000** | **0.9892** | **1.0000** | 1809.31 |

---

## Incremental Document Ingestion

The ingestion pipeline computes cryptographic content hashes (`SHA-256`) for every file:

```
Document Ingest
      │
      ▼
Compute SHA-256 Hash
      │
   Exists in Registry with identical hash?
   ├── YES ──> [Skip Processing (0ms IO)]
   └── NO  ──> [Purge Stale Chunks] ──> [Chunk & Re-embed] ──> [Update Index & Registry]
```

---

## Running the Unit Tests

```bash
python -m unittest discover tests
```

Output:
```
...............
----------------------------------------------------------------------
Ran 15 tests in 0.189s

OK
```

---

## Extensibility & Customization

### Using Google Gemini API

Set your environment variable:
```bash
export GEMINI_API_KEY="your-gemini-key"
python main.py
```

### Using Local Ollama

```python
from rag_pipeline import RAGPipeline
from core.llm import OllamaLLM

pipeline = RAGPipeline(llm=OllamaLLM(model_name="llama3:latest"))
result = pipeline.query("How does Raft consensus work?")
print(result.answer)
```

---

## License
MIT License. Built for educational rigor and high-performance production understanding.
