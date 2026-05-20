"""
Guardrails — validation and safety checks at every stage of the RAG pipeline.

Layers covered:
  1. FileGuardrails     — ingest-time: size, MIME type, filename, PII in content
  2. InputGuardrails    — query-time: length, prompt injection, PII in question
  3. RetrievalGuardrails— post-retrieval: relevance threshold, empty results
  4. OutputGuardrails   — post-generation: PII in answer, grounding check
  5. PIIDetector        — shared regex scanner used by all layers

Each check returns a GuardrailResult (passed, violations, warnings) so the
caller decides whether to block or just warn — no silent failures.
"""

import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    violations: List[str] = field(default_factory=list)   # hard blocks
    warnings: List[str]   = field(default_factory=list)   # soft alerts

    def merge(self, other: "GuardrailResult") -> "GuardrailResult":
        return GuardrailResult(
            passed=self.passed and other.passed,
            violations=self.violations + other.violations,
            warnings=self.warnings   + other.warnings,
        )


# ── PII detector (shared) ─────────────────────────────────────────────────────

class PIIDetector:
    """
    Regex-based PII scanner.
    For production, replace with Microsoft Presidio or AWS Comprehend.
    """

    _PATTERNS: Dict[str, str] = {
        "Email":       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "Phone (US)":  r"\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "Credit Card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
        "IP Address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "Passport":    r"\b[A-Z]{1,2}\d{6,9}\b",
        "Aadhaar":     r"\b\d{4}\s\d{4}\s\d{4}\b",          # Indian national ID
        "PAN Card":    r"\b[A-Z]{5}\d{4}[A-Z]\b",            # Indian tax ID
    }

    def scan(self, text: str) -> List[str]:
        """Return list of PII type names found in text."""
        found = []
        for pii_type, pattern in self._PATTERNS.items():
            if re.search(pattern, text):
                found.append(pii_type)
        return found

    def redact(self, text: str) -> str:
        """Replace PII occurrences with [REDACTED-<type>] markers."""
        for pii_type, pattern in self._PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED-{pii_type.upper().replace(' ', '_')}]", text)
        return text


_pii = PIIDetector()


# ── 1. File guardrails ────────────────────────────────────────────────────────

class FileGuardrails:
    MAX_SIZE_MB = 20
    ALLOWED_EXTENSIONS = {
        ".txt", ".md", ".pdf", ".docx", ".doc",
        ".xlsx", ".xls", ".csv", ".pptx",
    }

    # Magic bytes for binary format verification (extension spoofing prevention)
    _MAGIC: List[tuple] = [
        (b"%PDF",           {".pdf"}),
        (b"PK\x03\x04",    {".docx", ".xlsx", ".pptx", ".doc", ".xls"}),  # ZIP-based Office
        (b"\xd0\xcf\x11\xe0", {".doc", ".xls"}),                           # OLE2 legacy Office
    ]

    def validate(
        self,
        file_path: str,
        original_filename: str,
        file_size_bytes: int,
    ) -> GuardrailResult:
        result = GuardrailResult(passed=True)
        ext = Path(original_filename).suffix.lower()

        # 1. Filename sanitization — checked FIRST: path traversal is a critical
        #    security issue regardless of extension and must not be skipped.
        if ".." in original_filename or original_filename.startswith("/"):
            result.passed = False
            result.violations.append(
                f"Filename '{original_filename}' contains unsafe path characters."
            )
            return result   # stop immediately — do not touch this file

        # 2. Extension whitelist
        if ext not in self.ALLOWED_EXTENSIONS:
            result.passed = False
            result.violations.append(
                f"File type '{ext}' is not allowed. "
                f"Permitted: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )
            return result   # no point reading a disallowed file

        # 3. File size
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > self.MAX_SIZE_MB:
            result.passed = False
            result.violations.append(
                f"File size {size_mb:.1f} MB exceeds limit of {self.MAX_SIZE_MB} MB."
            )

        # 4. MIME / magic-byte check for binary formats
        if ext in {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx"}:
            magic_ok = self._check_magic(file_path, ext)
            if not magic_ok:
                result.passed = False
                result.violations.append(
                    f"File content does not match its '{ext}' extension "
                    f"(possible spoofing or corruption)."
                )

        return result

    def _check_magic(self, file_path: str, ext: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
        except OSError:
            return False
        for magic, allowed_exts in self._MAGIC:
            if header.startswith(magic) and ext in allowed_exts:
                return True
        # Plain-text formats have no magic bytes — always pass
        return ext in {".txt", ".md", ".csv"}

    def scan_content_pii(self, text: str) -> GuardrailResult:
        """Warn (not block) when ingested content contains PII."""
        result = GuardrailResult(passed=True)
        found = _pii.scan(text)
        if found:
            result.warnings.append(
                f"Document contains potential PII: {', '.join(found)}. "
                "Consider reviewing before ingesting into the knowledge base."
            )
        return result


# ── 2. Input guardrails ───────────────────────────────────────────────────────

class InputGuardrails:
    MAX_QUERY_LENGTH = 500
    MIN_QUERY_LENGTH = 3

    # Prompt injection patterns
    _INJECTION_PATTERNS: List[str] = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"forget\s+(your|all)\s+(instructions|context|rules|system)",
        r"you\s+are\s+now\s+(a|an)",
        r"act\s+as\s+(if\s+you\s+are|a|an)",
        r"jailbreak",
        r"system\s*prompt",
        r"override\s+(your|the)\s+(instructions|rules|context)",
        r"disregard\s+(your|all|the|previous)",
        r"do\s+anything\s+now",
        r"dan\s+mode",
        r"pretend\s+(you\s+are|to\s+be)",
        r"new\s+instructions\s*:",
        r"<\s*/?system\s*>",           # XML-style injection
        r"\[inst\]|\[\/inst\]",          # Llama [INST] markers (matched after lowercasing)
    ]

    def validate(self, query: str) -> GuardrailResult:
        result = GuardrailResult(passed=True)
        stripped = query.strip()

        # 1. Empty / too short
        if len(stripped) < self.MIN_QUERY_LENGTH:
            result.passed = False
            result.violations.append(
                f"Query is too short (minimum {self.MIN_QUERY_LENGTH} characters)."
            )
            return result

        # 2. Length cap
        if len(stripped) > self.MAX_QUERY_LENGTH:
            result.passed = False
            result.violations.append(
                f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters "
                f"(got {len(stripped)})."
            )

        # 3. Prompt injection detection
        lower = stripped.lower()
        for pattern in self._INJECTION_PATTERNS:
            if re.search(pattern, lower):
                result.passed = False
                result.violations.append(
                    "Query contains a prompt injection pattern and cannot be processed."
                )
                break

        # 4. PII in query (warn only — user may legitimately search for their own data)
        found = _pii.scan(stripped)
        if found:
            result.warnings.append(
                f"Your query appears to contain PII ({', '.join(found)}). "
                "Avoid including personal information in search queries."
            )

        return result


# ── 3. Retrieval guardrails ───────────────────────────────────────────────────

class RetrievalGuardrails:
    # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite.
    # Chunks above this threshold are considered irrelevant.
    RELEVANCE_THRESHOLD = 0.75

    def validate(self, hits: List[Dict[str, Any]]) -> GuardrailResult:
        result = GuardrailResult(passed=True)

        if not hits:
            result.passed = False
            result.violations.append(
                "No documents found in the knowledge base. "
                "Please ingest documents before querying."
            )
            return result

        # Filter out irrelevant chunks
        relevant = [h for h in hits if h["distance"] <= self.RELEVANCE_THRESHOLD]
        if not relevant:
            result.passed = False
            result.violations.append(
                "No sufficiently relevant content found for your question. "
                f"All retrieved chunks exceeded the relevance threshold "
                f"(distance > {self.RELEVANCE_THRESHOLD}). "
                "Try rephrasing or ensure the topic is covered in your documents."
            )
        elif len(relevant) < len(hits):
            result.warnings.append(
                f"{len(hits) - len(relevant)} chunk(s) were below the relevance "
                f"threshold and excluded from context."
            )

        return result

    def filter_relevant(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only chunks within the relevance threshold."""
        return [h for h in hits if h["distance"] <= self.RELEVANCE_THRESHOLD]


# ── 4. Output guardrails ──────────────────────────────────────────────────────

class OutputGuardrails:
    MIN_GROUNDING_WORDS = 2   # answer must share at least N words with the context

    def validate(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> GuardrailResult:
        result = GuardrailResult(passed=True)

        if not answer or not answer.strip():
            result.passed = False
            result.violations.append("Model returned an empty answer.")
            return result

        # 1. PII leakage in output
        found = _pii.scan(answer)
        if found:
            result.warnings.append(
                f"Answer may contain PII ({', '.join(found)}). "
                "Review before sharing externally."
            )

        # 2. Grounding check — verify answer shares content words with retrieved context
        context_text = " ".join(c["text"].lower() for c in context_chunks)
        answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
        context_words = set(re.findall(r"\b[a-z]{4,}\b", context_text))
        overlap = answer_words & context_words

        if len(overlap) < self.MIN_GROUNDING_WORDS:
            result.warnings.append(
                "Answer may not be grounded in the retrieved documents. "
                "Verify the response against the source chunks."
            )

        return result


# ── convenience singletons ────────────────────────────────────────────────────

file_guardrails      = FileGuardrails()
input_guardrails     = InputGuardrails()
retrieval_guardrails = RetrievalGuardrails()
output_guardrails    = OutputGuardrails()
pii_detector         = PIIDetector()
