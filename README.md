# RAG From First Principles

> **Production-oriented RAG internals implemented from first principles in Python.**  
> Core retrieval algorithms, vector indexing, BM25, hybrid rank fusion, citation grounding, and evaluation are engineered from first principles; pretrained models are used only for dense embedding representations and neural cross-encoder inference.

---

## 🎯 Why I Built This

Most Retrieval-Augmented Generation (RAG) applications abstract retrieval behind monolithic frameworks like LangChain or LlamaIndex, treating similarity search and ranking as black boxes.

This project implements the core retrieval and generation pipeline from first principles to understand and control every layer of the RAG lifecycle:
- How **recursive chunking and boundary preservation** impact embedding semantics and retrieval recall.
- How **dense vector geometry (NumPy)** compares against **sparse term matching (Okapi BM25)** across different query types.
- How **rank fusion (RRF)** combines disparate score distributions without heuristic calibration.
- How **two-stage retrieval with cross-encoder reranking** eliminates false positives.
- How to **quantitatively evaluate retrieval quality** using standard Information Retrieval metrics (Recall@K, Precision@K, MRR, NDCG@K).

---

## 🏗️ Architecture

```
                         ┌─────────────────────────┐
                         │    Source Documents     │
                         │    (.txt, .md, .pdf)    │
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
             Recall@K             Precision@K             MRR / NDCG@K
```

---

## 🧩 Key Features

| Component | What It Does | Implementation |
|---|---|---|
| **Document Loaders** | Ingests `.txt`, `.md`, `.pdf` with encoding detection and page metadata | [`core/loaders.py`](./core/loaders.py) |
| **Recursive Chunking** | Hierarchical boundary splitting (`\n\n` &rarr; `\n` &rarr; `. `) with sliding overlap | [`core/chunker.py`](./core/chunker.py) |
| **Dense Vector Store** | In-memory store with Cosine, Dot product, Euclidean distance, and metadata filters in pure NumPy | [`core/vector_store.py`](./core/vector_store.py) |
| **BM25 Inverted Index** | Okapi BM25 sparse keyword search from scratch ($k_1=1.5, b=0.75$, Robertson-Spärck Jones IDF) | [`core/bm25.py`](./core/bm25.py) |
| **Hybrid Retrieval** | Reciprocal Rank Fusion (RRF with $k=60$) and linear score normalization | [`core/retriever.py`](./core/retriever.py) |
| **Neural Reranking** | Cross-Encoder joint query-document cross-attention for precision reordering | [`core/reranker.py`](./core/reranker.py) |
| **CO-STAR Prompting** | Structured prompt assembly with anti-hallucination guardrails and inline citations (`[Source X]`) | [`core/prompt.py`](./core/prompt.py) |
| **Incremental Indexing** | SHA-256 document hashing to skip unchanged files and re-index modified content | [`storage/index.py`](./storage/index.py) |
| **Retrieval Evaluation** | Quantitative IR metrics: Recall@K, Precision@K, HitRate@K, MRR, NDCG@K | [`evaluation/metrics.py`](./evaluation/metrics.py) |

---

## 🔍 Retrieval Strategies

### 1. Dense Retrieval
- **Mechanism**: Projects query and chunks into continuous vector space; calculates Cosine similarity via NumPy matrix multiplication.
- **Best for**: Conceptual questions, semantic paraphrasing, and thematic relevance.

### 2. Sparse Retrieval (Okapi BM25)
- **Mechanism**: Token frequency normalized by document length and inverse document frequency.
- **Best for**: Exact keywords, technical identifiers (e.g. `CR2`, `xmin`), acronyms (e.g. `PBFT`, `2PL`), and error codes.

### 3. Hybrid Retrieval & RRF
- **Mechanism**: Queries both dense and sparse indices independently; fuses candidate rank lists using **Reciprocal Rank Fusion**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + r_m(d)}$$
- **Best for**: Production workloads where queries contain both conceptual descriptions and exact technical tokens.

### 4. Cross-Encoder Reranking
- **Mechanism**: Evaluates all-to-all cross-attention between full query and candidate text to resolve subtle semantic nuances and eliminate false positives.

---

## 📊 Retrieval Evaluation & Benchmark

Retrieval quality is measured quantitatively using a ground-truth benchmark dataset ([`data/eval_dataset.json`](./data/eval_dataset.json)) across multi-domain engineering documents (Distributed Consensus, Database Concurrency, OS Virtual Memory, Neural Optimization, Cloud Policies, Event Streaming).

### Measured Performance Comparison

| Retriever | Recall@5 | Precision@5 | MRR | NDCG@5 | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| **BM25 (Sparse)** | 0.9444 | 0.4333 | 1.0000 | 0.9609 | **0.07** |
| **Dense (NumPy)** | 0.9444 | 0.4333 | 1.0000 | 0.9507 | 46.85 |
| **Hybrid (RRF)** | **1.0000** | **0.4667** | **1.0000** | **0.9946** | 40.71 |
| **Hybrid + Reranker** | **1.0000** | **0.4667** | **1.0000** | **1.0000** | 1469.80 |

### Key Findings:
1. **Hybrid RRF achieves 100% Recall@5 (+5.56% over single retrievers)**: Merging dense semantics with sparse keyword matches eliminates individual retrieval blind spots.
2. **Cross-Encoder Reranking achieves a perfect 1.0000 NDCG@5**: Joint cross-attention places the single most authoritative context chunk at Rank 1 for 100% of benchmark queries.
3. **Sub-millisecond Lexical Latency**: The custom BM25 index returns candidates in **0.07ms**, making it suitable for ultra-fast first-stage candidate retrieval.

To reproduce these metrics:
```bash
poetry run rag benchmark
# or: python main.py benchmark
```

---

## 💡 End-to-End Example

### 1. Multi-Stage Retrieval Inspection (`/inspect`)

```text
Query: "What is the refund period and policy for enterprise customers?"
----------------------------------------------------------------------
1. cloud_platform_policy.txt (chunk: 0, id: b77ae61ea38d17d0_0)
   Dense Score:  0.5963     BM25 Score:   7.9226
   RRF Score:    0.03279    Rerank Score: 4.4228
   "# Cloud Platform Service Level Agreement & Refund Policies
    ## 1. Refund and Billing Policy
    Enterprise customers are eligible for a full refund within a 30-day trial period..."

2. cloud_platform_policy.txt (chunk: 1, id: b77ae61ea38d17d0_1)
   Dense Score:  0.3729     BM25 Score:   2.5303
   RRF Score:    0.03226    Rerank Score: -6.5446
   "To request a refund or credit, the organization's billing administrator must submit..."
```

### 2. Grounded Generation with Inline Citations

```text
=== Grounded Answer ===
Enterprise customers are eligible for a full refund within a 30-day trial period from the initial invoice date [Source 1]. Refund or credit requests submitted between 30 and 60 days are eligible for prorated service credits only, submitted via tickets tagged 'Billing Dispute' [Source 1, Source 2].

=== Sources & Citations ===
+------------+---------------------------+-------+---------------------------------------------+
| Tag        | Source Document           | Chunk | Preview Snippet                             |
+------------+---------------------------+-------+---------------------------------------------+
| [Source 1] | cloud_platform_policy.txt | 0     | # Cloud Platform SLA & Refund Policies...   |
| [Source 2] | cloud_platform_policy.txt | 1     | To request a refund or credit, submit a...  |
+------------+---------------------------+-------+---------------------------------------------+
```

---

## 📐 Design Decisions

- **Why implement BM25 from scratch?**  
  To understand inverted indexing, term frequency saturation curves ($k_1$), and length normalization penalties ($b$) without relying on Lucene, Elasticsearch, or external search servers.
- **Why vector search in pure NumPy?**  
  To explore exact matrix vector math (Cosine, Euclidean, Dot Product), partitioning algorithms, and persistence mechanics before introducing complex external vector databases.
- **Why Reciprocal Rank Fusion (RRF)?**  
  Dense similarity scores ($\in [-1, 1]$) and BM25 scores ($\in [0, \infty)$) have incompatible scales. RRF merges candidate positions rather than raw scores, avoiding fragile score calibration.
- **Why two-stage retrieval (Retrieve &rarr; Rerank)?**  
  Dense and sparse retrieval cast a wide net across thousands of chunks in milliseconds. The Cross-Encoder performs computationally expensive all-to-all attention on only the top 20–50 candidates.
- **Why separate retrieval from generation?**  
  Decoupling retrieval allows indexing and ranking quality to be quantitatively benchmarked (Recall@K, NDCG@K) independently from LLM prompt engineering.

---

## 📖 Deep Dive Documentation

Detailed architectural derivations and implementation notes:

- [**System Architecture & SOLID OOP Design**](./docs/overview.md)
- [**Phase 1: Ingestion, Loaders & Recursive Chunking**](./docs/phase1_ingestion_and_chunking.md)
- [**Phase 2: NumPy Dense Vector Store & Matrix Geometry**](./docs/phase2_dense_vector_store.md)
- [**Phase 3: Okapi BM25 Inverted Index from Scratch**](./docs/phase3_bm25_inverted_index.md)
- [**Phase 4: Hybrid Retrieval & Reciprocal Rank Fusion**](./docs/phase4_hybrid_retrieval_and_fusion.md)
- [**Phase 5: Cross-Encoder Neural Reranking**](./docs/phase5_cross_encoder_reranking.md)
- [**Phase 6: CO-STAR Prompting & Citation Grounding**](./docs/phase6_costar_prompt_and_generation.md)
- [**Phase 7: Evaluation Metrics & Automated Benchmarking**](./docs/phase7_evaluation_and_benchmarking.md)

---

## 📁 Repository Structure

```
core/
├── schema.py          # Document, Chunk, SearchResult, QueryResult
├── loaders.py         # Text, Markdown, PDF, Directory loaders
├── chunker.py         # Character & Recursive boundary splitters
├── embeddings.py      # SentenceTransformers & Gemini API embeddings
├── vector_store.py    # NumPy Dense Vector Store (Cosine / Dot / Euclidean)
├── bm25.py            # Okapi BM25 Inverted Index from scratch
├── retriever.py       # Dense, Sparse, and Hybrid (RRF) retrievers
├── reranker.py        # Cross-Encoder neural reranker
├── prompt.py          # CO-STAR prompt builder & citation grounding
└── llm.py             # Gemini, Ollama, and Mock LLM adapters

storage/
├── index.py           # Index manager & SHA-256 incremental hashing
└── persistence.py     # Atomic disk serialization

evaluation/
├── dataset.py         # Evaluation dataset schema & loaders
├── metrics.py         # Pure math: Recall@K, Precision@K, MRR, NDCG@K
├── evaluator.py       # Multi-query pipeline evaluator
└── benchmark.py       # Side-by-side comparative benchmark runner

data/
├── sample_knowledge/  # Curated technical knowledge base
└── eval_dataset.json  # Ground-truth evaluation dataset

tests/                 # Complete unit test suite (15 passing tests)
rag_pipeline.py        # End-to-end RAG pipeline orchestrator
main.py                # Interactive CLI & REPL
pyproject.toml         # Poetry package configuration
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Shikhaar/Rag-from-Scratch.git
cd Rag-from-Scratch
```

### 2. Install Dependencies

Using **Poetry**:
```bash
poetry install
```

Or using **pip**:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure LLM API Key

To use Google Gemini for generation:
```bash
# On Linux/macOS
export GEMINI_API_KEY="your_api_key_here"

# On Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"
```
*(If no API key is provided, the system automatically uses the built-in deterministic `MockLLM` for offline execution.)*

### 4. Run the Interactive CLI

```bash
# Using Poetry
poetry run rag

# Or using Python
python main.py
```

### CLI Commands:
- `/query <question>`: Run grounded generation with inline citations.
- `/inspect <question>`: Inspect multi-stage scores (`Dense`, `BM25`, `RRF`, `Rerank`).
- `/benchmark`: Execute the 4-way evaluation benchmark across the test dataset.
- `/ingest <path>`: Incrementally index a document or directory.
- `/mode <hybrid|dense|bm25>`: Switch active retrieval strategy.
- `/stats`: View active index statistics.

---

## 🧪 Testing

Run the automated unit test suite:

```bash
python -m unittest discover tests
```

Output:
```text
...............
----------------------------------------------------------------------
Ran 15 tests in 0.196s

OK
```

---

## 📜 License

MIT License. Designed for deep educational understanding and high-performance production engineering.
