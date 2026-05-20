"""
Output guardrail tests.

Two attack surfaces at generation time:
  1. PII leakage — the model echoes sensitive data from documents into the answer.
  2. Hallucination / ungrounded answer — model makes up content not in context,
     potentially delivering false or manipulated information.
"""

import pytest
from rag.guardrails import OutputGuardrails

guard = OutputGuardrails()

GROUNDED_CONTEXT = [
    {"text": "The company policy mandates annual security audits for all systems.",
     "source": "policy.pdf", "page": "3", "distance": 0.2},
    {"text": "Employees must complete compliance training every twelve months.",
     "source": "policy.pdf", "page": "4", "distance": 0.3},
]


# ── Empty answer ──────────────────────────────────────────────────────────────

class TestEmptyAnswer:
    def test_empty_string_blocked(self):
        r = guard.validate("", GROUNDED_CONTEXT)
        assert not r.passed
        assert any("empty" in v.lower() for v in r.violations)

    def test_whitespace_only_blocked(self):
        r = guard.validate("   \n  ", GROUNDED_CONTEXT)
        assert not r.passed


# ── PII leakage in answer ─────────────────────────────────────────────────────

class TestPIILeakage:
    def test_email_in_answer_warns(self):
        answer = "According to the policy, contact admin@company.com for support."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert r.passed          # not a hard block — caller decides
        assert any("PII" in w for w in r.warnings)

    def test_ssn_in_answer_warns(self):
        answer = "The employee record shows SSN 123-45-6789."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert r.passed
        assert any("PII" in w for w in r.warnings)

    def test_credit_card_in_answer_warns(self):
        answer = "Payment card 4111 1111 1111 1111 was charged."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert r.passed
        assert any("PII" in w for w in r.warnings)

    def test_phone_in_answer_warns(self):
        answer = "Call 555-123-4567 to reach the compliance officer."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert r.passed
        assert any("PII" in w for w in r.warnings)

    def test_multiple_pii_types_all_flagged(self):
        answer = "Email user@corp.com or call 555-123-4567."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert any("PII" in w for w in r.warnings)

    def test_clean_answer_no_pii_warning(self):
        answer = "Annual security audits are required for all systems."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert not any("PII" in w for w in r.warnings)


# ── Hallucination / grounding check ──────────────────────────────────────────

class TestGrounding:
    def test_grounded_answer_passes(self):
        """Answer shares vocabulary with the retrieved context."""
        answer = "According to the policy, annual security audits are mandatory."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert not any("grounded" in w.lower() for w in r.warnings)

    def test_ungrounded_answer_warns(self):
        """
        Answer is completely fabricated — shares no words with context.
        This is the hallucination attack: model invents information.
        """
        answer = "Jupiter has 95 known moons orbiting it."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert any("grounded" in w.lower() for w in r.warnings)

    def test_hallucinated_instructions_warn(self):
        """Attacker's goal: inject false policy via a hallucinated answer."""
        answer = "All passwords should be shared with the administrator immediately."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert any("grounded" in w.lower() for w in r.warnings)

    def test_empty_context_causes_grounding_warning(self):
        """
        No context at all → model has nothing to ground on → warning raised.
        """
        answer = "The policy says annual audits are required."
        r = guard.validate(answer, [])
        assert r.passed   # not a hard block
        assert any("grounded" in w.lower() for w in r.warnings)

    def test_partial_grounding_passes(self):
        """Answer shares some words with context — considered grounded."""
        answer = "Employees must complete the required compliance training."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert not any("grounded" in w.lower() for w in r.warnings)


# ── Clean answer with clean context ──────────────────────────────────────────

class TestCleanOutput:
    def test_fully_clean_answer_no_warnings(self):
        answer = "Annual security audits are mandatory for compliance."
        r = guard.validate(answer, GROUNDED_CONTEXT)
        assert r.passed
        assert not r.violations
        assert not r.warnings
