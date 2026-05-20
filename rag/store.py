"""
VectorStore — a thin wrapper around ChromaDB.

ChromaDB persists its data to disk (./chroma_db/) so your ingested documents
survive between runs. Each document chunk is stored with:
  - its embedding vector (for similarity search)
  - its text (returned at query time)
  - metadata: source filename, page number, chunk index
"""

import os
import uuid
from typing import List, Dict, Any

from .exceptions import StoreError, StoreUnavailableError

COLLECTION_NAME = "rag_documents"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class VectorStore:
    def __init__(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=PERSIST_DIR)
            self._col = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError as exc:
            raise StoreUnavailableError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc
        except Exception as exc:
            raise StoreUnavailableError(
                f"Cannot initialise the vector store at '{PERSIST_DIR}': {exc}. "
                "The directory may be corrupted or you may lack write permission."
            ) from exc

    # ── write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> int:
        """Store chunks with their embeddings and metadata. Returns count added."""
        if not chunks:
            raise StoreError("Cannot add an empty list of chunks.")
        if len(chunks) != len(embeddings) or len(chunks) != len(metadata):
            raise StoreError(
                f"Lengths must match: chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        ids = [str(uuid.uuid4()) for _ in chunks]
        try:
            self._col.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
            )
        except OSError as exc:
            raise StoreError(
                f"Disk error while writing to vector store: {exc}. "
                "Check available disk space."
            ) from exc
        except Exception as exc:
            raise StoreError(f"Failed to store chunks: {exc}") from exc

        return len(chunks)

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self, embedding: List[float], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the n_results most similar chunks to the given query embedding.
        Returns list of {text, source, page, distance}.
        """
        if not embedding:
            raise StoreError("Query embedding must not be empty.")

        count = self._safe_count()
        if count == 0:
            return []

        try:
            results = self._col.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise StoreError(f"Vector search failed: {exc}") from exc

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "page":     meta.get("page", "?"),
                "distance": round(dist, 4),
            })
        return hits

    def list_sources(self) -> List[str]:
        """Return unique source filenames currently in the store."""
        if self._safe_count() == 0:
            return []
        try:
            all_meta = self._col.get(include=["metadatas"])["metadatas"]
            return sorted(set(m.get("source", "?") for m in all_meta))
        except Exception as exc:
            raise StoreError(f"Failed to list sources: {exc}") from exc

    def count(self) -> int:
        return self._safe_count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        try:
            import chromadb
            self._client.delete_collection(COLLECTION_NAME)
            self._col = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StoreError(f"Failed to clear the knowledge base: {exc}") from exc

    # ── internal ──────────────────────────────────────────────────────────────

    def _safe_count(self) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0
