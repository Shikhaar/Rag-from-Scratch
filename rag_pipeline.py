"""
RAGPipeline: The end-to-end orchestrator for ingestion, hybrid retrieval, reranking, and grounded generation.
"""

import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

from core.schema import Document, Chunk, SearchResult, QueryResult
from core.loaders import TextLoader, MarkdownLoader, PDFLoader, DirectoryLoader
from core.chunker import BaseChunker, RecursiveCharacterChunker
from core.embeddings import BaseEmbeddings, SentenceTransformerEmbeddings, HashingEmbeddings
from core.vector_store import NumpyVectorStore
from core.bm25 import BM25Index
from core.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from core.reranker import BaseReranker, CrossEncoderReranker, PassthroughReranker
from core.prompt import PromptBuilder
from core.llm import BaseLLM, MockLLM, GeminiLLM, OllamaLLM
from storage.index import IndexManager

RetrievalMode = Literal["hybrid", "dense", "sparse", "bm25"]


class RAGPipeline:
    """
    Complete, self-contained RAG system built from first principles.
    Combines dense vector search, BM25 sparse keyword search, RRF fusion, cross-encoder reranking,
    citation grounding, and LLM generation.
    """

    def __init__(
        self,
        embeddings: Optional[BaseEmbeddings] = None,
        llm: Optional[BaseLLM] = None,
        reranker: Optional[BaseReranker] = None,
        chunker: Optional[BaseChunker] = None,
        index_dir: str = "./.rag_index",
        use_neural_models: bool = True,
    ):
        self.index_dir = os.path.abspath(index_dir)

        # 1. Initialize Embeddings
        if embeddings is not None:
            self.embeddings = embeddings
        elif use_neural_models:
            try:
                self.embeddings = SentenceTransformerEmbeddings()
            except Exception as e:
                print(f"[Warning] Falling back to HashingEmbeddings: {e}")
                self.embeddings = HashingEmbeddings()
        else:
            self.embeddings = HashingEmbeddings()

        # 2. Initialize Chunker
        self.chunker = chunker or RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)

        # 3. Initialize Storage & Index Manager
        self.index_manager = IndexManager(
            embeddings=self.embeddings,
            chunker=self.chunker,
            index_dir=self.index_dir,
        )

        # 4. Initialize Retrievers
        self.dense_retriever = DenseRetriever(self.index_manager.vector_store, self.embeddings)
        self.sparse_retriever = SparseRetriever(self.index_manager.bm25_index)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever,
            self.sparse_retriever,
            fusion_method="rrf",
            rrf_k=60,
        )

        # 5. Initialize Reranker
        if reranker is not None:
            self.reranker = reranker
        elif use_neural_models:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                print(f"[Warning] Falling back to PassthroughReranker: {e}")
                self.reranker = PassthroughReranker()
        else:
            self.reranker = PassthroughReranker()

        # 6. Initialize Prompt Builder & LLM
        self.prompt_builder = PromptBuilder()
        self.llm = llm or MockLLM()

    @property
    def vector_store(self) -> NumpyVectorStore:
        return self.index_manager.vector_store

    @property
    def bm25_index(self) -> BM25Index:
        return self.index_manager.bm25_index

    @property
    def chunks(self) -> List[Chunk]:
        return self.vector_store.chunks

    def ingest_file(self, file_path: str, force_reindex: bool = False) -> Dict[str, int]:
        """Ingest a single document file (.txt, .md, .pdf)."""
        file_path = os.path.abspath(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".py", ".json", ".csv"]:
            loader = TextLoader(file_path)
        elif ext in [".md", ".markdown"]:
            loader = MarkdownLoader(file_path)
        elif ext == ".pdf":
            loader = PDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        stats = self.index_manager.index_documents(docs, force_reindex=force_reindex)
        self.save()
        return stats

    def ingest_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        force_reindex: bool = False,
    ) -> Dict[str, int]:
        """Ingest all supported documents in a directory incrementally."""
        loader = DirectoryLoader(dir_path, recursive=recursive)
        docs = loader.load()
        stats = self.index_manager.index_documents(docs, force_reindex=force_reindex)
        self.save()
        return stats

    def ingest_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        force_reindex: bool = False,
    ) -> Dict[str, int]:
        """Ingest raw in-memory strings with optional metadata."""
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if (metadatas and i < len(metadatas)) else {"source": f"text_{i}"}
            docs.append(Document(page_content=text, metadata=meta))

        stats = self.index_manager.index_documents(docs, force_reindex=force_reindex)
        self.save()
        return stats

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        candidate_k: int = 20,
    ) -> List[SearchResult]:
        """
        Execute search across dense, sparse, or hybrid retrieval with optional reranking.
        """
        fetch_k = candidate_k if (rerank and self.reranker) else top_k

        if mode == "dense":
            results = self.dense_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        elif mode in ["sparse", "bm25"]:
            results = self.sparse_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        elif mode == "hybrid":
            results = self.hybrid_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Apply Cross-Encoder Reranking
        if rerank and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def inspect(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        candidate_k: int = 20,
    ) -> List[SearchResult]:
        """
        Retrieve candidates and populate ALL granular stage scores
        (Dense Score, BM25 Score, RRF Score, and Rerank Score) for transparent inspection.
        """
        # Step 1: Query dense vectors
        q_vec = self.embeddings.embed_query(query)
        dense_candidates = self.vector_store.search(q_vec, top_k=candidate_k, filters=filters)
        dense_map = {r.chunk.chunk_id: r.score for r in dense_candidates}

        # Step 2: Query BM25 index
        bm25_candidates = self.bm25_index.search(query, top_k=candidate_k, filters=filters)
        bm25_map = {r.chunk.chunk_id: r.score for r in bm25_candidates}

        # Step 3: Compute RRF Hybrid scores
        hybrid_results = self.hybrid_retriever.retrieve(query, top_k=candidate_k, filters=filters)
        rrf_map = {r.chunk.chunk_id: r.score for r in hybrid_results}

        # Ensure dense & bm25 scores are attached
        for r in hybrid_results:
            cid = r.chunk.chunk_id
            r.dense_score = dense_map.get(cid)
            r.bm25_score = bm25_map.get(cid)
            r.rrf_score = rrf_map.get(cid)

        # Step 4: Cross-Encoder Rerank
        if self.reranker and hybrid_results:
            final_results = self.reranker.rerank(query, hybrid_results, top_k=top_k)
        else:
            final_results = hybrid_results[:top_k]

        return final_results

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> QueryResult:
        """
        Full RAG lifecycle: Retrieval -> Reranking -> Grounded Prompt Construction -> LLM Generation.
        """
        start_time = time.perf_counter()

        # Step 1 & 2: Retrieve and Rerank
        retrieved_chunks = self.search(
            query=query_text,
            top_k=top_k,
            mode=mode,
            filters=filters,
            rerank=rerank,
        )

        # Step 3: Format Grounded Prompt and Context
        sys_prompt, user_prompt, citations = self.prompt_builder.build_prompt(
            query=query_text,
            results=retrieved_chunks,
        )

        # Step 4: LLM Generation
        answer = self.llm.generate(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return QueryResult(
            query=query_text,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            metadata={
                "retrieval_mode": mode,
                "filters": filters,
                "reranked": rerank,
                "top_k": top_k,
            },
            latency_ms=latency_ms,
        )

    def save(self, directory_path: Optional[str] = None) -> None:
        """Save index state to disk."""
        target_dir = directory_path or self.index_dir
        self.index_manager.save(target_dir)

    def load(self, directory_path: Optional[str] = None) -> None:
        """Load index state from disk."""
        target_dir = directory_path or self.index_dir
        self.index_manager.load(target_dir)
        # Reconnect retrievers to restored stores
        self.dense_retriever.vector_store = self.index_manager.vector_store
        self.sparse_retriever.bm25_index = self.index_manager.bm25_index

    def stats(self) -> Dict[str, Any]:
        """Return system statistics."""
        return {
            "total_documents": len(self.index_manager.registry),
            "total_chunks": len(self.vector_store),
            "embedding_dimension": self.embeddings.dimension,
            "vector_store_metric": self.vector_store.metric,
            "bm25_vocabulary_size": len(self.bm25_index.inverted_index),
            "bm25_avg_doc_len": round(self.bm25_index.avg_doc_len, 2),
            "reranker_type": type(self.reranker).__name__,
            "index_directory": self.index_dir,
            "registered_documents": list(self.index_manager.registry.keys()),
        }
