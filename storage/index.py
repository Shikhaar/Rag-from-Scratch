"""
Unified Index Manager with Incremental Document Hashing and Deduplication.
"""

import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from core.schema import Document, Chunk
from core.chunker import BaseChunker, RecursiveCharacterChunker
from core.embeddings import BaseEmbeddings
from core.vector_store import NumpyVectorStore
from core.bm25 import BM25Index
from storage.persistence import atomic_write_json, load_json


class IndexManager:
    """
    Orchestrates dense vector indexing and BM25 sparse indexing with incremental caching.
    Avoids re-embedding or re-indexing unmodified documents.
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        chunker: Optional[BaseChunker] = None,
        vector_store: Optional[NumpyVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        index_dir: Optional[str] = None,
    ):
        self.embeddings = embeddings
        self.chunker = chunker or RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
        self.vector_store = vector_store or NumpyVectorStore(metric="cosine")
        self.bm25_index = bm25_index or BM25Index()
        self.index_dir = index_dir or os.path.abspath("./.rag_index")

        # Document registry: doc_id -> {source, doc_hash, chunk_count, chunk_ids, modified_at, indexed_at}
        self.registry: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.index_dir):
            self.load()

    def _registry_path(self) -> str:
        return os.path.join(self.index_dir, "registry.json")

    def _load_registry(self) -> None:
        if os.path.exists(self._registry_path()):
            self.registry = load_json(self._registry_path(), default={})

    def _save_registry(self) -> None:
        atomic_write_json(self._registry_path(), self.registry)

    def _hash_document_content(self, document: Document) -> str:
        raw = f"{document.metadata.get('source', '')}:{document.page_content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def index_documents(
        self,
        documents: List[Document],
        force_reindex: bool = False,
    ) -> Dict[str, int]:
        """
        Incrementally index a batch of documents.
        Returns summary statistics of {indexed: N, skipped: M, total_chunks: K}.
        """
        # Check if existing vector dimension matches current embedding model
        if self.vector_store.dimension is not None and self.vector_store.dimension != self.embeddings.dimension:
            force_reindex = True
            self.vector_store = NumpyVectorStore(metric=self.vector_store.metric)
            self.bm25_index = BM25Index()
            self.registry = {}

        indexed_docs = 0
        skipped_docs = 0
        new_chunks: List[Chunk] = []

        for doc in documents:
            doc_id = doc.doc_id or "unknown"
            source = doc.metadata.get("source", "unknown")
            content_hash = self._hash_document_content(doc)
            modified_at = doc.metadata.get("modified_at", time.time())

            # Check if document already exists and is unchanged
            if not force_reindex and doc_id in self.registry:
                existing = self.registry[doc_id]
                if existing.get("doc_hash") == content_hash:
                    skipped_docs += 1
                    continue

            # If document was modified, delete old version first
            if doc_id in self.registry:
                self.delete_document(doc_id)

            # Chunk document
            chunks = self.chunker.split_document(doc)
            if not chunks:
                skipped_docs += 1
                continue

            new_chunks.extend(chunks)
            indexed_docs += 1

            # Update registry
            self.registry[doc_id] = {
                "source": source,
                "filename": doc.metadata.get("filename", os.path.basename(source)),
                "doc_hash": content_hash,
                "chunk_count": len(chunks),
                "chunk_ids": [c.chunk_id for c in chunks],
                "modified_at": modified_at,
                "indexed_at": time.time(),
            }

        # Embed and add new chunks
        if new_chunks:
            chunk_texts = [c.content for c in new_chunks]
            embeddings_arr = self.embeddings.embed_documents(chunk_texts)

            self.vector_store.add_chunks(new_chunks, embeddings_arr)
            self.bm25_index.add_chunks(new_chunks)

        self._save_registry()

        return {
            "indexed_documents": indexed_docs,
            "skipped_documents": skipped_docs,
            "new_chunks": len(new_chunks),
            "total_chunks": len(self.vector_store),
            "total_documents": len(self.registry),
        }

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from all indices and registry."""
        if doc_id not in self.registry:
            return False

        # Remove from vector store
        self.vector_store.delete_document(doc_id)

        # Rebuild BM25 index from remaining vector store chunks
        new_bm25 = BM25Index(k1=self.bm25_index.k1, b=self.bm25_index.b)
        new_bm25.add_chunks(self.vector_store.chunks)
        self.bm25_index = new_bm25

        del self.registry[doc_id]
        self._save_registry()
        return True

    def save(self, directory_path: Optional[str] = None) -> None:
        """Persist full index state (vectors, bm25, metadata, registry)."""
        target_dir = directory_path or self.index_dir
        os.makedirs(target_dir, exist_ok=True)

        self.vector_store.save(os.path.join(target_dir, "dense_store"))
        self.bm25_index.save(os.path.join(target_dir, "sparse_index"))
        atomic_write_json(os.path.join(target_dir, "registry.json"), self.registry)

    def load(self, directory_path: Optional[str] = None) -> None:
        """Load index state from disk."""
        target_dir = directory_path or self.index_dir
        dense_path = os.path.join(target_dir, "dense_store")
        sparse_path = os.path.join(target_dir, "sparse_index")
        reg_path = os.path.join(target_dir, "registry.json")

        if os.path.exists(dense_path):
            self.vector_store = NumpyVectorStore.load(dense_path)
        if os.path.exists(sparse_path):
            self.bm25_index = BM25Index.load(sparse_path)
        if os.path.exists(reg_path):
            self.registry = load_json(reg_path, default={})
