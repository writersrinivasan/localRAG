"""
Embedder — turns text into vectors using a local sentence-transformers model.

Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Fast, lightweight, good quality for semantic search
  - Downloaded automatically on first use (~90 MB)
"""

from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        # Loaded once; subsequent calls reuse the same model in memory
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors, one per input text."""
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
