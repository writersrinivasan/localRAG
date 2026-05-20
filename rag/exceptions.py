"""
Custom exception hierarchy for localRAG.

Every module raises its own typed subclass of RAGError so callers
can catch broadly (RAGError) or precisely (EmbedderTimeoutError).
"""


class RAGError(Exception):
    """Base class for all localRAG errors."""


# ── loader ────────────────────────────────────────────────────────────────────

class LoaderError(RAGError):
    """Raised when a document cannot be parsed or yields no text."""


# ── chunker ───────────────────────────────────────────────────────────────────

class ChunkerError(RAGError):
    """Raised when chunking parameters are invalid."""


# ── embedder ──────────────────────────────────────────────────────────────────

class EmbedderError(RAGError):
    """Raised when embedding fails (model error, bad input, etc.)."""


class EmbedderTimeoutError(EmbedderError):
    """Raised when the embedding call exceeds its timeout."""


# ── vector store ──────────────────────────────────────────────────────────────

class StoreError(RAGError):
    """Raised when ChromaDB cannot read or write data."""


class StoreUnavailableError(StoreError):
    """Raised when the ChromaDB collection cannot be initialised."""


# ── generator ─────────────────────────────────────────────────────────────────

class GeneratorError(RAGError):
    """Raised when the LLM returns an unusable response."""


class GeneratorTimeoutError(GeneratorError):
    """Raised when the local model inference exceeds its timeout."""


class GeneratorAPIError(GeneratorError):
    """Raised when the remote API returns an error."""
