"""
Retrieval guardrail tests.

An attacker may try to:
  - Query an empty store (no documents) and trick the system into generating
    a hallucinated answer from zero context.
  - Craft a query whose embedding lands far from all stored chunks so that
    irrelevant chunks are retrieved and fed to the generator.

The relevance threshold (cosine distance > 0.75) blocks both scenarios.
"""

import pytest
from rag.guardrails import RetrievalGuardrails
from tests.conftest import make_hit

guard = RetrievalGuardrails()


# ── Empty store attack ────────────────────────────────────────────────────────

class TestEmptyStore:
    def test_no_hits_blocked(self):
        """System has no documents — must not generate anything."""
        r = guard.validate([])
        assert not r.passed
        assert any("knowledge base" in v.lower() for v in r.violations)

    def test_violation_message_is_actionable(self):
        r = guard.validate([])
        assert r.violations      # at least one violation


# ── Irrelevance / out-of-scope attack ─────────────────────────────────────────

class TestIrrelevanceAttack:
    def test_all_chunks_above_threshold_blocked(self):
        """
        All retrieved chunks are semantically unrelated (distance > 0.75).
        Attacker hopes the model hallucinates an answer from irrelevant context.
        """
        hits = [
            make_hit(0.80),
            make_hit(0.90),
            make_hit(0.85),
        ]
        r = guard.validate(hits)
        assert not r.passed
        assert any("threshold" in v.lower() or "relevant" in v.lower() for v in r.violations)

    def test_exactly_at_threshold_blocked(self):
        """Distance == 0.76 is above 0.75 → blocked."""
        hits = [make_hit(0.76)]
        r = guard.validate(hits)
        assert not r.passed

    def test_boundary_distance_blocked(self):
        """Distance == 1.0 (orthogonal vectors) → completely irrelevant."""
        hits = [make_hit(1.0), make_hit(0.99)]
        r = guard.validate(hits)
        assert not r.passed

    def test_maximum_distance_blocked(self):
        """Distance == 2.0 (opposite vectors) → maximum irrelevance."""
        hits = [make_hit(2.0)]
        r = guard.validate(hits)
        assert not r.passed


# ── Partial relevance (mixed hits) ───────────────────────────────────────────

class TestPartialRelevance:
    def test_mixed_hits_passes_with_warning(self):
        """
        Some chunks are relevant, some are not.
        System should pass (use the relevant ones) but warn about excluded chunks.
        """
        hits = [
            make_hit(0.30),   # relevant
            make_hit(0.50),   # relevant
            make_hit(0.85),   # irrelevant — should be excluded
        ]
        r = guard.validate(hits)
        assert r.passed
        assert r.warnings     # warned about excluded chunk

    def test_warning_mentions_excluded_count(self):
        hits = [make_hit(0.20), make_hit(0.90), make_hit(0.95)]
        r = guard.validate(hits)
        assert r.passed
        assert any("2" in w for w in r.warnings)   # 2 chunks excluded


# ── Relevant hits pass cleanly ────────────────────────────────────────────────

class TestRelevantHits:
    def test_single_relevant_chunk_passes(self):
        r = guard.validate([make_hit(0.30)])
        assert r.passed
        assert not r.violations

    def test_all_relevant_passes_no_warning(self):
        hits = [make_hit(0.10), make_hit(0.40), make_hit(0.70)]
        r = guard.validate(hits)
        assert r.passed
        assert not r.warnings

    def test_exactly_at_threshold_passes(self):
        """Distance == 0.75 is at (not above) the threshold → passes."""
        r = guard.validate([make_hit(0.75)])
        assert r.passed

    def test_very_similar_chunk_passes(self):
        """Distance == 0.01 means near-identical — should always pass."""
        r = guard.validate([make_hit(0.01)])
        assert r.passed


# ── filter_relevant ───────────────────────────────────────────────────────────

class TestFilterRelevant:
    def test_filters_out_irrelevant(self):
        hits = [make_hit(0.30), make_hit(0.80), make_hit(0.50)]
        relevant = guard.filter_relevant(hits)
        assert len(relevant) == 2
        assert all(h["distance"] <= 0.75 for h in relevant)

    def test_empty_input_returns_empty(self):
        assert guard.filter_relevant([]) == []

    def test_all_irrelevant_returns_empty(self):
        hits = [make_hit(0.80), make_hit(0.90)]
        assert guard.filter_relevant(hits) == []

    def test_all_relevant_returns_all(self):
        hits = [make_hit(0.10), make_hit(0.50), make_hit(0.70)]
        assert guard.filter_relevant(hits) == hits
