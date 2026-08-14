# System Architecture & OOP Design Overview

`RAG From First Principles` is a modular, extensible Retrieval-Augmented Generation system designed strictly according to Object-Oriented Programming (SOLID) design patterns in Python without external framework dependencies.

---

## 🏛️ System Architecture

```
                          ┌─────────────────────────┐
                          │   Source Documents      │
                          │   (.txt, .md, .pdf)     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   BaseLoader Subclasses │
                          │ (Text, Markdown, PDF)   │
                          └────────────┬────────────┘
                                       │
                                   Documents
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   BaseChunker Hierarchy │
                          │ (Recursive / Character) │
                          └────────────┬────────────┘
                                       │
                                Chunk + Metadata
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
              BaseEmbeddings                  BaseInvertedIndex
          (SentenceTransformers/API)         (Okapi BM25 Index)
                       │                               │
                       ▼                               ▼
                BaseVectorStore                        │
              (NumPy Vector Store)                     │
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                               [ User Query ]
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
                 DenseRetriever                  SparseRetriever
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                                HybridRetriever
                             (RRF / Score Blending)
                                       ▼
                               Top-N Candidates
                                       ▼
                                 BaseReranker
                           (CrossEncoderReranker)
                                       ▼
                               Top-K Reranked
                                       ▼
                               BasePromptBuilder
                            (COSTARPromptBuilder)
                                       ▼
                                    BaseLLM
                            (Gemini / Ollama / Mock)
                                       ▼
                           Grounded Answer + Citations
                                       │
                                       ▼
                          Evaluation & Benchmark Suite
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
              Recall@K                MRR                NDCG@K
```

---

## 💎 SOLID Object-Oriented Design

### 1. Single Responsibility Principle (SRP)
- `BaseLoader` implementations handle file extraction and encoding detection.
- `BaseChunker` implementations handle boundary splitting and metadata enrichment.
- `BaseVectorStore` handles geometric vector operations and dense indexing.
- `BaseInvertedIndex` handles lexical tokenization, term frequency indexing, and BM25 scoring.
- `BaseRetriever` implementations handle candidate search.
- `BaseReranker` implementations handle candidate precision re-scoring.
- `BasePromptBuilder` handles structured CO-STAR prompt synthesis.
- `BaseLLM` handles generative inference.
- `IndexManager` coordinates incremental caching and deduplication.

### 2. Open/Closed Principle (OCP)
The system is open for extension but closed for modification:
- To add a new document type (e.g. Docx or HTML), subclass `BaseLoader`.
- To use an alternate embedding model (e.g. OpenAI or Voyage), subclass `BaseEmbeddings`.
- To introduce a new vector database engine, subclass `BaseVectorStore`.
- To swap the reranking model, subclass `BaseReranker`.

### 3. Liskov Substitution Principle (LSP)
Every subclass can be substituted anywhere its parent ABC is expected. For example:
- `HashingEmbeddings` and `SentenceTransformerEmbeddings` can be swapped transparently in `RAGPipeline` without altering any retrieval logic.
- `CrossEncoderReranker` and `PassthroughReranker` conform to the identical `rerank(query, results, top_k)` interface.

### 4. Interface Segregation Principle (ISP)
Interfaces are concise, typed, and focused. Data models (`Document`, `Chunk`, `SearchResult`, `QueryResult`) use Python dataclasses with clear properties.

### 5. Dependency Inversion Principle (DIP)
High-level orchestrators (`RAGPipeline`, `IndexManager`, `RetrievalEvaluator`, `RAGBenchmark`) depend strictly on abstractions (`BaseEmbeddings`, `BaseRetriever`, `BaseReranker`, `BaseLLM`), injected via standard constructor parameters.
