"""
Okapi BM25 Sparse Keyword Search Index implemented from first principles.
Mathematical formulation: Robertson-Spärck Jones IDF with Okapi term saturation.
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Set
from core.schema import Chunk, SearchResult

# Standard English stopwords
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class BaseInvertedIndex(ABC):
    """Abstract Base Class for sparse lexical keyword inverted indices."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def save(self, directory_path: str) -> None:
        pass


class BM25Index(BaseInvertedIndex):
    """
    Okapi BM25 inverted index for fast and accurate sparse lexical retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
    ):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Document length normalization parameter
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

        self.chunks: List[Chunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0

        # Inverted index: term -> {chunk_index: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # Document frequencies: term -> count of docs containing term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        # Precomputed IDFs: term -> idf_score
        self.idf_cache: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric terms, stripping stopwords."""
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return [t for t in raw_tokens if t not in self.stopwords and len(t) > 1]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index a list of chunks into the BM25 inverted index."""
        if not chunks:
            return

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)

            tokens = self.tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)

            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                self.inverted_index[term][idx] = freq
                self.doc_frequencies[term] += 1

        self.num_docs = len(self.chunks)
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.num_docs)
        self._recompute_idf()

    def _recompute_idf(self) -> None:
        """
        Compute non-negative Okapi IDF for each unique vocabulary term:
        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
        """
        self.idf_cache = {}
        N = self.num_docs

        for term, n_q in self.doc_frequencies.items():
            # Standard smoothed BM25 IDF formulation
            numerator = N - n_q + 0.5
            denominator = n_q + 0.5
            idf = math.log((numerator / denominator) + 1.0)
            self.idf_cache[term] = max(1e-6, idf)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Score and rank chunks against query using BM25 formula.
        """
        if self.num_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Accumulate BM25 scores across matching documents
        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf_cache.get(token, 0.0)
            posting_list = self.inverted_index[token]

            for doc_idx, tf in posting_list.items():
                # Length normalization component
                doc_len = self.doc_lengths[doc_idx]
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)

                # Okapi TF term
                tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                scores[doc_idx] += idf * tf_component

        # Apply metadata filters
        filtered_scores = []
        for doc_idx, score in scores.items():
            chunk = self.chunks[doc_idx]
            if filters:
                match = True
                for k, v in filters.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            filtered_scores.append((doc_idx, score))

        # Sort top candidates descending
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = filtered_scores[:top_k]

        results: List[SearchResult] = []
        for rank, (doc_idx, score) in enumerate(top_matches, 1):
            chunk = self.chunks[doc_idx]
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=float(score),
                    rank=rank,
                )
            )

        return results

    def save(self, directory_path: str) -> None:
        """Persist BM25 index and vocab statistics to disk."""
        os.makedirs(directory_path, exist_ok=True)
        save_path = os.path.join(directory_path, "bm25_index.json")

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

        # Convert int keys in inverted index to str for JSON serialization
        json_inverted = {
            term: {str(doc_idx): tf for doc_idx, tf in postings.items()}
            for term, postings in self.inverted_index.items()
        }

        data = {
            "k1": self.k1,
            "b": self.b,
            "num_docs": self.num_docs,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_frequencies": self.doc_frequencies,
            "idf_cache": self.idf_cache,
            "inverted_index": json_inverted,
            "chunks": serializable_chunks,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, directory_path: str) -> "BM25Index":
        """Load BM25 index from disk."""
        save_path = os.path.join(directory_path, "bm25_index.json")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"BM25 index not found at {save_path}")

        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls(k1=data["k1"], b=data["b"])
        idx.num_docs = data["num_docs"]
        idx.avg_doc_len = data["avg_doc_len"]
        idx.doc_lengths = data["doc_lengths"]
        idx.doc_frequencies = defaultdict(int, data["doc_frequencies"])
        idx.idf_cache = data["idf_cache"]

        # Restore inverted index with int keys
        idx.inverted_index = defaultdict(
            dict,
            {
                term: {int(doc_idx): tf for doc_idx, tf in postings.items()}
                for term, postings in data["inverted_index"].items()
            },
        )

        # Restore chunks
        for item in data["chunks"]:
            chunk = Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                content=item["content"],
                metadata=item["metadata"],
                chunk_index=item["chunk_index"],
                chunk_hash=item.get("chunk_hash", ""),
            )
            idx.chunks.append(chunk)

        return idx
