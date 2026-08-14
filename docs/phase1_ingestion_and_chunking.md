# Phase 1: Document Ingestion & Chunking

Phase 1 establishes the input boundary of the RAG system: transforming raw files (`.txt`, `.md`, `.pdf`) into structured `Document` objects and splitting them into semantically coherent `Chunk` representations.

---

## 1. Document Schema & Loaders

### `Document` Data Model
```python
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
```
- **Deterministic ID Generation**: SHA-256 hash of `source + page_content` to uniquely identify document revisions.
- **Metadata Fields**: `source`, `filename`, `file_type`, `file_size`, `modified_at`, `page`, `encoding`.

### Loader Implementations
- **`TextLoader`**: Multi-encoding fallback handler (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
- **`MarkdownLoader`**: Preserves markdown document structure and headers.
- **`PDFLoader`**: Uses `pypdf` to extract text page-by-page, attaching `page` and `total_pages` metadata.
- **`DirectoryLoader`**: Recursively discovers supported files and delegates to the appropriate specialized loader.

---

## 2. Text Splitting & Chunking Strategies

### The Chunking Dilemma
- **Too small**: Loss of surrounding context, leading to incoherent embedding semantics.
- **Too large**: Diluted semantic density, exceeding embedding model token limits, and introducing irrelevant context.

### `RecursiveCharacterChunker`
Splits text hierarchically using natural language boundaries before falling back to character-level splits:

$$\text{Separators} = [\text{"\\n\\n"}, \text{"\\n"}, \text{". "}, \text{"? "}, \text{"! "}, \text{" "}, \text{""}]$$

```
Raw Text
   │
   ├── Try splitting by "\n\n" (Paragraphs)
   │     │
   │     ├── If piece < chunk_size ──> Merge with adjacent pieces up to chunk_size
   │     └── If piece >= chunk_size ──> Recurse to "\n" (Sentences/Lines)
   │                                      │
   │                                      ├── If piece >= chunk_size ──> Recurse to ". " (Sentence boundaries)
   │                                      └── If piece >= chunk_size ──> Recurse to " " (Words)
```

### Sliding Window Overlap
When a chunk boundary is reached, `chunk_overlap` characters are retained from the tail of the previous chunk, ensuring that sentences straddling boundary edges are not bisected or lost.

---

## 3. Incremental Indexing & Hashing

```
File Modification Time / Content
               │
               ▼
      Compute SHA-256 Hash
               │
      Exists in Registry?
      ├── YES and Hash Unchanged ──> Skip Processing (0ms IO)
      └── NO or Hash Changed     ──> Delete Old Chunks ──> Split & Re-Index
```
The `IndexManager` persists a document registry (`registry.json`) recording `doc_hash`, `chunk_count`, `modified_at`, and `indexed_at` timestamps.
