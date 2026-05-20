"""
Loader safety tests.

Verifies that malicious or malformed files are rejected cleanly
rather than causing crashes or silent data loss.
"""

import os
import stat
import pytest
from rag.loader import load_file
from rag.exceptions import LoaderError


# ── Non-existent and inaccessible files ───────────────────────────────────────

class TestMissingFiles:
    def test_nonexistent_file_raises(self):
        with pytest.raises(LoaderError, match="not found"):
            load_file("/tmp/this_file_does_not_exist_xyz.txt")

    def test_directory_as_file_raises(self, tmp_path):
        with pytest.raises(LoaderError):
            load_file(str(tmp_path))   # tmp_path is a directory, not a file

    def test_empty_file_raises(self, tmp_path):
        """Zero-byte file must raise — nothing to ingest."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(LoaderError, match="empty"):
            load_file(str(f))


# ── Unsupported file types ────────────────────────────────────────────────────

class TestUnsupportedTypes:
    def test_exe_raises(self, tmp_path):
        f = tmp_path / "attack.exe"
        f.write_bytes(b"MZ\x90\x00")
        with pytest.raises(LoaderError, match="Unsupported"):
            load_file(str(f))

    def test_py_raises(self, tmp_path):
        f = tmp_path / "evil.py"
        f.write_text("import os")
        with pytest.raises(LoaderError, match="Unsupported"):
            load_file(str(f))

    def test_no_extension_raises(self, tmp_path):
        f = tmp_path / "noext"
        f.write_text("data")
        with pytest.raises(LoaderError, match="Unsupported"):
            load_file(str(f))


# ── Corrupted / wrong-format files ───────────────────────────────────────────

class TestCorruptedFiles:
    def test_fake_pdf_raises(self, tmp_path):
        """File named .pdf but containing plain text — parser will fail."""
        f = tmp_path / "fake.pdf"
        f.write_text("this is not a pdf")
        with pytest.raises(LoaderError):
            load_file(str(f))

    def test_fake_docx_raises(self, tmp_path):
        """File named .docx but not a valid ZIP/Office document."""
        f = tmp_path / "fake.docx"
        f.write_text("not a docx")
        with pytest.raises(LoaderError):
            load_file(str(f))

    def test_fake_xlsx_raises(self, tmp_path):
        f = tmp_path / "fake.xlsx"
        f.write_bytes(b"not an xlsx")
        with pytest.raises(LoaderError):
            load_file(str(f))

    def test_truncated_binary_raises(self, tmp_path):
        """Only first 4 bytes of a PDF — truncated/corrupt."""
        f = tmp_path / "truncated.pdf"
        f.write_bytes(b"%PDF")   # valid header, nothing else
        with pytest.raises(LoaderError):
            load_file(str(f))


# ── Text file correctness ─────────────────────────────────────────────────────

class TestTextLoader:
    def test_plain_text_loaded(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello, this is a test document about security.")
        pages = load_file(str(f))
        assert len(pages) == 1
        assert "security" in pages[0]["text"]
        assert pages[0]["source"] == "doc.txt"
        assert pages[0]["page"] == 1

    def test_markdown_loaded(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nContent paragraph here.")
        pages = load_file(str(f))
        assert len(pages) == 1
        assert "Title" in pages[0]["text"]

    def test_source_filename_set(self, tmp_path):
        f = tmp_path / "report.txt"
        f.write_text("Quarterly report data.")
        pages = load_file(str(f))
        assert pages[0]["source"] == "report.txt"

    def test_encoding_errors_handled(self, tmp_path):
        """Files with invalid UTF-8 bytes must not crash — use errors=replace."""
        f = tmp_path / "latin.txt"
        f.write_bytes(b"Caf\xe9 au lait is a French drink.")
        pages = load_file(str(f))
        assert len(pages) == 1   # loaded, not crashed


# ── CSV correctness ───────────────────────────────────────────────────────────

class TestCSVLoader:
    def test_valid_csv_loaded(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age,city\nAlice,30,NY\nBob,25,LA")
        pages = load_file(str(f))
        assert len(pages) == 1
        assert "Alice" in pages[0]["text"]

    def test_empty_csv_raises(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        with pytest.raises(LoaderError):
            load_file(str(f))

    def test_headers_only_csv_raises(self, tmp_path):
        """CSV with only a header row and no data rows."""
        f = tmp_path / "headers.csv"
        f.write_text("name,age,city\n")
        with pytest.raises(LoaderError):
            load_file(str(f))


# ── Permission denied ─────────────────────────────────────────────────────────

class TestPermissions:
    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses permission checks")
    def test_unreadable_file_raises(self, tmp_path):
        f = tmp_path / "secret.txt"
        f.write_text("secret data")
        f.chmod(0o000)   # remove all permissions
        try:
            with pytest.raises(LoaderError, match="[Pp]ermission"):
                load_file(str(f))
        finally:
            f.chmod(0o644)   # restore so tmp_path cleanup works
