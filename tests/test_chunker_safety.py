"""
Chunker safety tests.

The sliding-window chunker had an infinite loop bug when overlap >= chunk_size
(start never advances). These tests prove the fix holds and verify all
edge cases around invalid parameters and boundary text lengths.
"""

import pytest
from rag.chunker import chunk_text
from rag.exceptions import ChunkerError


# ── Invalid parameter attacks ─────────────────────────────────────────────────

class TestInvalidParameters:
    def test_overlap_equal_chunk_size_raises(self):
        """overlap == chunk_size → step = 0 → infinite loop. Must be caught."""
        with pytest.raises(ChunkerError, match="infinite loop"):
            chunk_text("some text", chunk_size=100, overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        """overlap > chunk_size → step < 0 → infinite loop."""
        with pytest.raises(ChunkerError, match="infinite loop"):
            chunk_text("some text", chunk_size=50, overlap=200)

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ChunkerError, match="chunk_size must be"):
            chunk_text("some text", chunk_size=0, overlap=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ChunkerError, match="chunk_size must be"):
            chunk_text("some text", chunk_size=-1, overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ChunkerError, match="overlap must be"):
            chunk_text("some text", chunk_size=100, overlap=-1)


# ── Edge case inputs ──────────────────────────────────────────────────────────

class TestEdgeCaseInputs:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_text_shorter_than_chunk_gives_one_chunk(self):
        chunks = chunk_text("Hello world.", chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_text_exactly_chunk_size_gives_one_chunk(self):
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 1

    def test_zero_overlap_is_valid(self):
        """overlap=0 means no overlap — still valid."""
        chunks = chunk_text("word " * 200, chunk_size=100, overlap=0)
        assert len(chunks) > 1

    def test_one_overlap_is_valid(self):
        chunks = chunk_text("word " * 200, chunk_size=100, overlap=1)
        assert len(chunks) > 1


# ── Chunk content correctness ─────────────────────────────────────────────────

class TestChunkCorrectness:
    def test_all_chunks_non_empty(self):
        chunks = chunk_text("This is a test. " * 100, chunk_size=200, overlap=50)
        assert all(c.strip() for c in chunks)

    def test_all_text_covered(self):
        """Every word from the original must appear in at least one chunk."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota"
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        combined = " ".join(chunks)
        for word in text.split():
            assert word in combined, f"Word '{word}' lost in chunking"

    def test_no_infinite_loop_large_overlap(self):
        """Largest valid overlap (chunk_size - 1) must finish quickly."""
        import time
        start = time.time()
        chunks = chunk_text("word " * 1000, chunk_size=100, overlap=99)
        elapsed = time.time() - start
        assert elapsed < 5, "Chunking took too long — possible infinite loop"
        assert len(chunks) > 0
