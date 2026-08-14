"""
Document loaders for ingesting various file formats into Document objects.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Optional
from core.schema import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data and return a list of Document objects."""
        pass


class TextLoader(BaseLoader):
    """Loads plain text files with automatic encoding handling."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = os.path.abspath(file_path)
        self.encoding = encoding

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        encodings = [self.encoding, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None
        used_enc = None

        for enc in encodings:
            try:
                with open(self.file_path, "r", encoding=enc) as f:
                    content = f.read()
                    used_enc = enc
                    break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise ValueError(f"Could not decode file {self.file_path} with any attempted encoding.")

        stat = os.stat(self.file_path)
        metadata = {
            "source": self.file_path,
            "filename": os.path.basename(self.file_path),
            "file_type": "text",
            "file_size": stat.st_size,
            "modified_at": stat.st_mtime,
            "encoding": used_enc,
        }
        return [Document(page_content=content, metadata=metadata)]


class MarkdownLoader(BaseLoader):
    """Loads Markdown documents with structural metadata."""

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        text_loader = TextLoader(self.file_path)
        docs = text_loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "markdown"
        return docs


class PDFLoader(BaseLoader):
    """Loads PDF files page-by-page with page-level metadata."""

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package is required for PDF loading. Please run `pip install pypdf`."
            )

        documents = []
        stat = os.stat(self.file_path)

        with open(self.file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                metadata = {
                    "source": self.file_path,
                    "filename": os.path.basename(self.file_path),
                    "file_type": "pdf",
                    "file_size": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "page": page_num + 1,
                    "total_pages": total_pages,
                }
                documents.append(Document(page_content=text, metadata=metadata))

        return documents


class DirectoryLoader(BaseLoader):
    """Recursively loads supported documents from a directory."""

    EXT_MAP = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".pdf": PDFLoader,
        ".py": TextLoader,
        ".json": TextLoader,
        ".csv": TextLoader,
    }

    def __init__(
        self,
        directory_path: str,
        recursive: bool = True,
        supported_extensions: Optional[List[str]] = None,
    ):
        self.directory_path = os.path.abspath(directory_path)
        self.recursive = recursive
        self.supported_extensions = (
            [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in supported_extensions]
            if supported_extensions
            else list(self.EXT_MAP.keys())
        )

    def load(self) -> List[Document]:
        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")

        documents: List[Document] = []
        p = Path(self.directory_path)
        pattern = "**/*" if self.recursive else "*"

        for file_path in p.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.supported_extensions and ext in self.EXT_MAP:
                    loader_cls = self.EXT_MAP[ext]
                    try:
                        loader = loader_cls(str(file_path))
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        print(f"[Warning] Failed to load {file_path}: {e}")

        return documents
