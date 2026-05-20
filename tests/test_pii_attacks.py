"""
PII detection tests.

Verifies that all 8 PII types are detected in both queries and document
content. Also verifies the redact() function masks values correctly and
that clean text produces no false positives.
"""

import pytest
from rag.guardrails import PIIDetector, InputGuardrails

pii    = PIIDetector()
guard  = InputGuardrails()


# ── Email ─────────────────────────────────────────────────────────────────────

class TestEmailDetection:
    def test_basic_email(self):
        assert "Email" in pii.scan("Contact us at user@example.com")

    def test_email_with_dots(self):
        assert "Email" in pii.scan("john.doe.smith@company.co.uk")

    def test_email_with_plus(self):
        assert "Email" in pii.scan("user+tag@domain.org")

    def test_email_in_sentence(self):
        assert "Email" in pii.scan("Please email admin@corp.com for access.")

    def test_no_email_in_clean_text(self):
        assert "Email" not in pii.scan("Send a message through the contact form.")


# ── US Phone ──────────────────────────────────────────────────────────────────

class TestPhoneDetection:
    def test_dashes(self):
        assert "Phone (US)" in pii.scan("Call 555-123-4567")

    def test_parentheses(self):
        assert "Phone (US)" in pii.scan("Reach us at (555) 123-4567")

    def test_with_country_code(self):
        assert "Phone (US)" in pii.scan("+1-555-123-4567")

    def test_dots_format(self):
        assert "Phone (US)" in pii.scan("555.123.4567")

    def test_no_phone_in_clean_text(self):
        assert "Phone (US)" not in pii.scan("The office opens at 9 and closes at 5.")


# ── SSN ───────────────────────────────────────────────────────────────────────

class TestSSNDetection:
    def test_standard_ssn(self):
        assert "SSN" in pii.scan("SSN: 123-45-6789")

    def test_ssn_inline(self):
        assert "SSN" in pii.scan("Employee 987-65-4321 terminated.")

    def test_no_ssn_in_date(self):
        """Date-like patterns (2023-01-15) should not match SSN (ddd-dd-dddd)."""
        assert "SSN" not in pii.scan("Report dated 2023-01-15 is ready.")


# ── Credit Card ───────────────────────────────────────────────────────────────

class TestCreditCardDetection:
    def test_spaces_format(self):
        assert "Credit Card" in pii.scan("Card: 4111 1111 1111 1111")

    def test_dashes_format(self):
        assert "Credit Card" in pii.scan("4111-1111-1111-1111")

    def test_no_spaces_format(self):
        assert "Credit Card" in pii.scan("4111111111111111")

    def test_no_cc_in_normal_numbers(self):
        """A 4-digit number repeated shouldn't trigger."""
        assert "Credit Card" not in pii.scan("Reference: 1234 items shipped.")


# ── IP Address ────────────────────────────────────────────────────────────────

class TestIPAddressDetection:
    def test_private_ip(self):
        assert "IP Address" in pii.scan("Server at 192.168.1.100")

    def test_public_ip(self):
        assert "IP Address" in pii.scan("External IP: 203.0.113.42")

    def test_loopback(self):
        assert "IP Address" in pii.scan("Localhost is 127.0.0.1")


# ── Passport ──────────────────────────────────────────────────────────────────

class TestPassportDetection:
    def test_us_passport_format(self):
        assert "Passport" in pii.scan("Passport: A12345678")

    def test_two_letter_prefix(self):
        assert "Passport" in pii.scan("Travel doc AB1234567")


# ── Aadhaar (Indian National ID) ──────────────────────────────────────────────

class TestAadhaarDetection:
    def test_aadhaar_standard(self):
        assert "Aadhaar" in pii.scan("Aadhaar: 1234 5678 9012")

    def test_aadhaar_inline(self):
        assert "Aadhaar" in pii.scan("ID verified: 9876 5432 1098")


# ── PAN Card (Indian Tax ID) ──────────────────────────────────────────────────

class TestPANCardDetection:
    def test_pan_standard(self):
        assert "PAN Card" in pii.scan("PAN: ABCDE1234F")

    def test_pan_inline(self):
        assert "PAN Card" in pii.scan("Tax ID PQRST5678Z submitted.")


# ── Redaction ─────────────────────────────────────────────────────────────────

class TestRedaction:
    def test_email_redacted(self):
        result = pii.redact("Contact user@example.com now.")
        assert "user@example.com" not in result
        assert "[REDACTED" in result

    def test_ssn_redacted(self):
        result = pii.redact("SSN is 123-45-6789.")
        assert "123-45-6789" not in result
        assert "[REDACTED" in result

    def test_multiple_pii_all_redacted(self):
        text = "Email user@test.com, SSN 123-45-6789, card 4111 1111 1111 1111."
        result = pii.redact(text)
        assert "user@test.com"       not in result
        assert "123-45-6789"         not in result
        assert "4111 1111 1111 1111" not in result

    def test_clean_text_unchanged(self):
        text = "The quarterly report shows 15% growth."
        assert pii.redact(text) == text


# ── PII in query triggers warning (not block) ─────────────────────────────────

class TestPIIInQuery:
    def test_email_in_query_warns_not_blocks(self):
        r = guard.validate("Find documents related to user@example.com")
        assert r.passed          # not blocked
        assert r.warnings        # warned

    def test_ssn_in_query_warns_not_blocks(self):
        r = guard.validate("What policy covers SSN 123-45-6789?")
        assert r.passed
        assert r.warnings

    def test_clean_query_no_warning(self):
        r = guard.validate("What are the data retention policies?")
        assert r.passed
        assert not r.warnings
