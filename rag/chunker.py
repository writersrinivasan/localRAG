"""
Text chunker — splits long text into overlapping windows.

Why overlap? So that a sentence split across two chunks doesn't lose context.
The overlap carries the tail of the previous chunk into the next one.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of ~chunk_size characters with overlap.

    chunk_size : target characters per chunk
    overlap    : characters shared between consecutive chunks
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Snap to the nearest sentence/word boundary so chunks don't cut mid-word
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # step back by overlap for the next window

    return chunks
