"""
Embedding wrappers for dense vector representations.
Supports local Sentence-Transformers, Google Gemini API, and lightweight pure-math fallback.
"""

from abc import ABC, abstractmethod
import math
import os
import re
from typing import List, Optional, Union
import numpy as np


class BaseEmbeddings(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of documents. Returns a 2D numpy array of shape (N, D)."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns a 1D numpy array of shape (D,)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension of the embeddings."""
        pass


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Local dense embeddings using HuggingFace Sentence-Transformers.
    Default model: 'all-MiniLM-L6-v2' (384 dimensions, fast & accurate).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install via `pip install sentence-transformers`."
            )

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class GeminiEmbeddings(BaseEmbeddings):
    """Google Gemini embedding model API wrapper (text-embedding-004)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Pass `api_key` or set GEMINI_API_KEY in environment."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
        except ImportError:
            raise ImportError("google-generativeai is required. Install via `pip install google-generativeai`.")

        self.model_name = model_name
        self._dimension = 768

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = []
        for text in texts:
            result = self.client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])

        arr = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        result = self.client.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
        )
        arr = np.array(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    @property
    def dimension(self) -> int:
        return self._dimension


class HashingEmbeddings(BaseEmbeddings):
    """
    Lightweight, deterministic feature hashing embedding built from pure math & NumPy.
    Zero external dependencies or downloads. Ideal for offline unit testing.
    """

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    def _hash_token(self, token: str) -> int:
        import zlib
        return zlib.crc32(token.encode("utf-8")) % self._dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not tokens:
            return vec

        for token in tokens:
            idx = self._hash_token(token)
            vec[idx] += 1.0

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._text_to_vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension
