"""
Core data structures for the RAG From First Principles system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import time


@dataclass
class Document:
    """Represents a raw ingested document with source metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            # Generate deterministic hash based on content and source
            source = self.metadata.get("source", "")
            raw = f"{source}:{self.page_content}"
            self.doc_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = time.time()


@dataclass
class Chunk:
    """Represents a text chunk created from a Document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    chunk_hash: str = ""

    def __post_init__(self):
        if not self.chunk_hash:
            self.chunk_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class SearchResult:
    """Represents a retrieved chunk with granular scoring details across stages."""
    chunk: Chunk
    score: float = 0.0
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: Optional[int] = None

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.metadata

    @property
    def source(self) -> str:
        return self.chunk.metadata.get("source", "unknown")

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class QueryResult:
    """Encapsulates the end-to-end response of the RAG pipeline."""
    query: str
    answer: str
    retrieved_chunks: List[SearchResult] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
