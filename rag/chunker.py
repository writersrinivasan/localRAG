"""
Text chunker — splits long text into overlapping windows.

Why overlap? So that a sentence split across two chunks doesn't lose context.
The overlap carries the tail of the previous chunk into the next one.
"""

from typing import List
from .exceptions import ChunkerError


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of ~chunk_size characters with overlap.

    chunk_size : target characters per chunk
    overlap    : characters shared between consecutive chunks
    """
    if chunk_size <= 0:
        raise ChunkerError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ChunkerError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ChunkerError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size}), "
            "otherwise the window never advances and creates an infinite loop."
        )

    text = text.strip()
    if not text:
        return []

    step = chunk_size - overlap   # guaranteed > 0 by the guard above
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Snap to the nearest word boundary so chunks don't cut mid-word
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += step

    return chunks
