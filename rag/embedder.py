"""
Embedder — turns text into vectors using a local sentence-transformers model.

Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Fast, lightweight, good quality for semantic search
  - Downloaded automatically on first use (~90 MB)
"""

import concurrent.futures
from typing import List

from .exceptions import EmbedderError, EmbedderTimeoutError

EMBED_TIMEOUT_SECONDS = 60


class Embedder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        except ImportError as exc:
            raise EmbedderError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbedderError(
                f"Failed to load embedding model '{self.MODEL_NAME}': {exc}. "
                "Check your internet connection or HuggingFace cache."
            ) from exc

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors, one per input text."""
        if not texts:
            raise EmbedderError("Cannot embed an empty list of texts.")
        if any(not isinstance(t, str) for t in texts):
            raise EmbedderError("All items to embed must be strings.")

        blank_count = sum(1 for t in texts if not t.strip())
        if blank_count == len(texts):
            raise EmbedderError("All texts are blank — nothing to embed.")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._model.encode, texts,
                    show_progress_bar=False,
                )
                result = future.result(timeout=EMBED_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise EmbedderTimeoutError(
                f"Embedding timed out after {EMBED_TIMEOUT_SECONDS} s. "
                "The model may be overloaded — try a smaller batch."
            )
        except Exception as exc:
            raise EmbedderError(f"Embedding failed: {exc}") from exc

        return result.tolist()

    def embed_one(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbedderError("Query text must be a non-empty string.")
        return self.embed([text])[0]
