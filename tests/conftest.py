"""
Shared fixtures for the localRAG security test suite.
"""

import os
import tempfile
import pytest

from rag.guardrails import (
    FileGuardrails,
    InputGuardrails,
    RetrievalGuardrails,
    OutputGuardrails,
    PIIDetector,
)


# ── guardrail fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def file_guard():
    return FileGuardrails()

@pytest.fixture
def input_guard():
    return InputGuardrails()

@pytest.fixture
def retrieval_guard():
    return RetrievalGuardrails()

@pytest.fixture
def output_guard():
    return OutputGuardrails()

@pytest.fixture
def pii():
    return PIIDetector()


# ── file helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def make_file(tmp_path):
    """Factory: create a temp file with given bytes, return its path."""
    created = []

    def _make(filename: str, content: bytes) -> str:
        path = tmp_path / filename
        path.write_bytes(content)
        created.append(str(path))
        return str(path)

    yield _make


@pytest.fixture
def txt_file(make_file):
    """A legitimate plain-text file."""
    return make_file("document.txt", b"This is a test document about RAG systems.")


# ── sample hit fixtures ───────────────────────────────────────────────────────

def make_hit(distance: float, text: str = "sample text about policy") -> dict:
    return {"text": text, "source": "doc.pdf", "page": "1", "distance": distance}
