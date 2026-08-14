"""
Unit tests for text chunkers (CharacterChunker and RecursiveCharacterChunker).
"""

import unittest
from core.schema import Document
from core.chunker import CharacterChunker, RecursiveCharacterChunker


class TestChunkers(unittest.TestCase):

    def test_character_chunker_sliding_window(self):
        doc = Document(page_content="0123456789" * 10, metadata={"source": "test.txt"})
        chunker = CharacterChunker(chunk_size=30, chunk_overlap=10)
        chunks = chunker.split_document(doc)

        self.assertTrue(len(chunks) > 1)
        for i, c in enumerate(chunks):
            self.assertTrue(len(c.content) <= 30)
            self.assertEqual(c.metadata["source"], "test.txt")
            self.assertEqual(c.chunk_index, i)

    def test_recursive_chunker_paragraph_preservation(self):
        text = "Paragraph 1 content here.\n\nParagraph 2 is separate.\n\nParagraph 3 is also long enough."
        doc = Document(page_content=text, metadata={"source": "doc.md"})
        chunker = RecursiveCharacterChunker(chunk_size=35, chunk_overlap=5)
        chunks = chunker.split_document(doc)

        self.assertTrue(len(chunks) >= 3)
        self.assertIn("Paragraph 1", chunks[0].content)
        self.assertEqual(chunks[0].metadata["source"], "doc.md")

    def test_empty_document(self):
        doc = Document(page_content="", metadata={})
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_document(doc)
        self.assertEqual(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
