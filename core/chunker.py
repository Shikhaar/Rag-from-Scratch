"""
Text chunking strategies implemented from first principles.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.schema import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def split_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into a list of Chunks."""
        pass

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of Documents into Chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class CharacterChunker(BaseChunker):
    """Fixed-size sliding window character chunker with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> List[Chunk]:
        text = document.page_content
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        chunk_idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]

            chunk_meta = document.metadata.copy()
            chunk_meta.update({
                "start_char": start,
                "end_char": end,
                "chunk_index": chunk_idx,
            })

            chunk_id = f"{document.doc_id}_{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_content,
                    metadata=chunk_meta,
                    chunk_index=chunk_idx,
                )
            )

            if end == len(text):
                break
            start += step
            chunk_idx += 1

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """
    Splits text recursively using a hierarchy of separators.
    Preserves semantic boundaries (paragraphs, sentences, words) before splitting arbitrarily.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the best matching separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        # Split text with chosen separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits up to chunk_size
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard split if no finer separators remain
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx:idx + self.chunk_size])

        # Merge adjacent pieces with overlap
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge split parts while respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        sep_len = len(separator) if self.keep_separator else 0

        for piece in splits:
            piece_len = len(piece)
            if total_len + piece_len + (sep_len if current_doc else 0) > self.chunk_size:
                if total_len > 0:
                    joined = separator.join(current_doc) if self.keep_separator else "".join(current_doc)
                    if joined.strip():
                        docs.append(joined.strip())

                    # Apply overlap: keep pieces from the tail until total length <= overlap
                    while total_len > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total_len -= len(popped) + sep_len

                current_doc.append(piece)
                total_len = sum(len(p) for p in current_doc) + max(0, len(current_doc) - 1) * sep_len
            else:
                current_doc.append(piece)
                total_len += piece_len + (sep_len if len(current_doc) > 1 else 0)

        if current_doc:
            joined = separator.join(current_doc) if self.keep_separator else "".join(current_doc)
            if joined.strip():
                docs.append(joined.strip())

        return docs

    def split_document(self, document: Document) -> List[Chunk]:
        text = document.page_content
        if not text:
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        char_offset = 0
        for idx, chunk_text in enumerate(raw_chunks):
            # Locate approximate character position in original text
            start_pos = text.find(chunk_text[:50], char_offset) if len(chunk_text) >= 50 else text.find(chunk_text, char_offset)
            if start_pos == -1:
                start_pos = char_offset
            end_pos = start_pos + len(chunk_text)
            char_offset = max(start_pos, 0)

            chunk_meta = document.metadata.copy()
            chunk_meta.update({
                "chunk_index": idx,
                "start_char": start_pos,
                "end_char": end_pos,
                "chunk_length": len(chunk_text),
            })

            chunk_id = f"{document.doc_id}_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_text,
                    metadata=chunk_meta,
                    chunk_index=idx,
                )
            )

        return chunks
