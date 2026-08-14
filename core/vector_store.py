"""
NumPy Dense Vector Store implemented from first principles.
Supports Cosine Similarity, Dot Product, Euclidean Distance, Metadata Filtering, and Disk Persistence.
"""

from abc import ABC, abstractmethod
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from core.schema import Chunk, SearchResult

MetricType = Literal["cosine", "dot_product", "euclidean"]


class BaseVectorStore(ABC):
    """Abstract Base Class defining the dense vector store contract."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> Optional[int]:
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def save(self, directory_path: str) -> None:
        pass


class NumpyVectorStore(BaseVectorStore):
    """
    In-memory dense vector store powered by NumPy matrix operations.
    No FAISS, Qdrant, or Pinecone - 100% mathematical implementation.
    """

    def __init__(self, metric: MetricType = "cosine"):
        self.metric = metric
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, D)
        self.chunk_id_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> Optional[int]:
        if self.vectors is not None and len(self.vectors) > 0:
            return self.vectors.shape[1]
        return None

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add text chunks and their dense embeddings into the store.
        `embeddings` must be a 2D numpy array of shape (len(chunks), D).
        """
        if len(chunks) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embeddings shape {embeddings.shape} does not match number of chunks {len(chunks)}"
            )

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = idx

        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks associated with a specific document ID."""
        if not self.chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        deleted_count = len(self.chunks) - len(keep_indices)

        if deleted_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            self.vectors = self.vectors[keep_indices] if len(keep_indices) > 0 else None
            self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        return deleted_count

    def _compute_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores between query vector and all stored vectors.
        Returns 1D array of scores of length N.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.empty((0,), dtype=np.float32)

        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 2 and query_vec.shape[0] == 1:
            query_vec = query_vec[0]

        if self.metric == "cosine":
            # Cosine Similarity: dot(q, v) / (||q|| * ||v||)
            dot_products = np.dot(self.vectors, query_vec)
            doc_norms = np.linalg.norm(self.vectors, axis=1)
            query_norm = np.linalg.norm(query_vec)

            denom = doc_norms * query_norm
            denom[denom == 0.0] = 1e-10
            scores = dot_products / denom

        elif self.metric == "dot_product":
            scores = np.dot(self.vectors, query_vec)

        elif self.metric == "euclidean":
            # Euclidean Distance: ||v - q||_2 -> mapped to similarity: 1 / (1 + dist)
            diff = self.vectors - query_vec
            dists = np.linalg.norm(diff, axis=1)
            scores = 1.0 / (1.0 + dists)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return scores

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search with optional metadata filtering.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        scores = self._compute_similarities(query_embedding)

        # Apply metadata filters if provided
        valid_indices = []
        for idx, chunk in enumerate(self.chunks):
            if filters:
                match = True
                for k, v in filters.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            if score_threshold is not None and scores[idx] < score_threshold:
                continue

            valid_indices.append(idx)

        if not valid_indices:
            return []

        valid_scores = scores[valid_indices]
        valid_indices = np.array(valid_indices)

        # Top-K sorting (descending order)
        if len(valid_scores) <= top_k:
            sorted_order = np.argsort(-valid_scores)
        else:
            # Efficient top-k partition + sort
            partition_idx = np.argpartition(-valid_scores, top_k)[:top_k]
            sorted_order = partition_idx[np.argsort(-valid_scores[partition_idx])]

        results: List[SearchResult] = []
        for rank, order_idx in enumerate(sorted_order, 1):
            original_idx = valid_indices[order_idx]
            chunk = self.chunks[original_idx]
            score = float(valid_scores[order_idx])

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    rank=rank,
                )
            )

        return results

    def save(self, directory_path: str) -> None:
        """Serialize vectors (.npz) and chunk metadata (.json) to disk."""
        os.makedirs(directory_path, exist_ok=True)

        vec_path = os.path.join(directory_path, "vectors.npz")
        meta_path = os.path.join(directory_path, "chunks_metadata.json")

        # Save vectors
        if self.vectors is not None:
            np.savez_compressed(vec_path, vectors=self.vectors)
        else:
            np.savez_compressed(vec_path, vectors=np.empty((0, 0), dtype=np.float32))

        # Save metadata and chunk contents
        serializable_chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "metadata": c.metadata,
                "chunk_index": c.chunk_index,
                "chunk_hash": c.chunk_hash,
            }
            for c in self.chunks
        ]

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": self.metric,
                    "count": len(self.chunks),
                    "chunks": serializable_chunks,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory_path: str) -> "NumpyVectorStore":
        """Load vector store from disk."""
        vec_path = os.path.join(directory_path, "vectors.npz")
        meta_path = os.path.join(directory_path, "chunks_metadata.json")

        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Store files not found in {directory_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        store = cls(metric=meta_data.get("metric", "cosine"))

        # Reconstruct chunks
        for item in meta_data["chunks"]:
            chunk = Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                content=item["content"],
                metadata=item["metadata"],
                chunk_index=item["chunk_index"],
                chunk_hash=item.get("chunk_hash", ""),
            )
            store.chunks.append(chunk)
            store.chunk_id_to_idx[chunk.chunk_id] = len(store.chunks) - 1

        # Load vectors
        data = np.load(vec_path)
        store.vectors = data["vectors"]
        if store.vectors.size == 0:
            store.vectors = None

        return store
