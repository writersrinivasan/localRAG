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
import chromadb


COLLECTION_NAME = "rag_documents"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=PERSIST_DIR)
        # get_or_create so repeated runs reuse existing data
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> int:
        """Store chunks with their embeddings and metadata. Returns count added."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._col.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
        )
        return len(chunks)

    # ── read ──────────────────────────────────────────────────────────────────

    def query(
        self, embedding: List[float], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the n_results most similar chunks to the given query embedding.
        Returns list of {text, source, page, distance}.
        """
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._col.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "distance": round(dist, 4),
            })
        return hits

    def list_sources(self) -> List[str]:
        """Return unique source filenames currently in the store."""
        if self._col.count() == 0:
            return []
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return sorted(set(m.get("source", "?") for m in all_meta))

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        """Delete all documents from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
