"""
Unit tests for NumPy Dense Vector Store (Cosine similarity, filtering, persistence).
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from core.schema import Chunk
from core.vector_store import NumpyVectorStore


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cosine_similarity_ranking(self):
        store = NumpyVectorStore(metric="cosine")

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Python AI machine learning", metadata={"category": "tech"}),
            Chunk(chunk_id="c2", doc_id="d1", content="Italian pizza pasta recipe", metadata={"category": "food"}),
            Chunk(chunk_id="c3", doc_id="d2", content="Deep learning neural networks", metadata={"category": "tech"}),
        ]

        # Artificial 3D vectors
        # c1: [1.0, 0.0, 0.0]
        # c2: [0.0, 1.0, 0.0]
        # c3: [0.9, 0.1, 0.0]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ], dtype=np.float32)

        store.add_chunks(chunks, embeddings)
        self.assertEqual(len(store), 3)

        # Query vector close to c1 and c3
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)
        self.assertEqual(results[1].chunk.chunk_id, "c3")

    def test_metadata_filtering(self):
        store = NumpyVectorStore(metric="cosine")
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", content="Text 1", metadata={"source": "a.txt", "page": 1}),
            Chunk(chunk_id="c2", doc_id="d1", content="Text 2", metadata={"source": "b.txt", "page": 2}),
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)

        # Query matching c2, but filter for source=a.txt
        query_vec = np.array([0.0, 1.0], dtype=np.float32)
        results = store.search(query_vec, top_k=5, filters={"source": "a.txt"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")

    def test_persistence_save_load(self):
        store = NumpyVectorStore(metric="cosine")
        chunks = [Chunk(chunk_id="c1", doc_id="d1", content="Saved content", metadata={"key": "val"})]
        embeddings = np.array([[0.5, 0.5]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)

        store.save(self.temp_dir)
        loaded = NumpyVectorStore.load(self.temp_dir)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.chunks[0].content, "Saved content")
        self.assertEqual(loaded.chunks[0].metadata["key"], "val")
        self.assertIsNotNone(loaded.vectors)


if __name__ == "__main__":
    unittest.main()
