"""
Audit logger — append-only JSONL audit trail for compliance.

Every ingest, query, generation, and guardrail event is logged to
audit.log as a JSON line. Each entry has a timestamp, event type,
and structured details. PII is never written to the log.

Rotation: 5 MB per file, keeps last 5 rotated files (audit.log.1 … .5).
If the log cannot be written (disk full, permission denied), the error
is printed to stderr — it never raises into the application.

Log location: <project_root>/audit.log
"""

import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH     = os.path.join(os.path.dirname(__file__), "..", "audit.log")
MAX_BYTES    = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 5                  # keep audit.log.1 … audit.log.5

# PII scrub applied to all values before writing
_PII_SCRUB = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"   # email
    r"|\b\d{3}-\d{2}-\d{4}\b"                                    # SSN
    r"|\b(?:\d{4}[\s\-]?){3}\d{4}\b"                            # credit card
    r"|\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"  # phone
)


def _scrub(value: str) -> str:
    return _PII_SCRUB.sub("[REDACTED]", str(value))


# ── rotating logger setup ─────────────────────────────────────────────────────

class _RawLineFormatter(logging.Formatter):
    """Passes the log message through without any modification."""
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("localrag.audit")
    if logger.handlers:
        return logger   # already configured (e.g. Streamlit reruns)
    try:
        handler = logging.handlers.RotatingFileHandler(
            LOG_PATH,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(_RawLineFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    except Exception as exc:
        print(f"[AUDIT] Cannot open log file '{LOG_PATH}': {exc}", file=sys.stderr)
    return logger


_logger = _build_logger()


def _write(entry: Dict[str, Any]) -> None:
    """Serialise entry to JSON and append to the rotating log. Never raises."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        line = json.dumps(entry)
        _logger.info(line)
    except Exception as exc:
        # Last-resort: at least surface it in the process output
        print(f"[AUDIT WRITE ERROR] {exc} — entry: {entry}", file=sys.stderr)


# ── public logging functions ──────────────────────────────────────────────────

def log_ingest(
    filename: str,
    size_kb: float,
    chunks_added: int,
    pii_types_found: List[str],
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    _write({
        "event":        "FILE_INGESTED",
        "file":         _scrub(filename),
        "size_kb":      round(size_kb, 1),
        "chunks_added": chunks_added,
        "pii_detected": bool(pii_types_found),
        "pii_types":    pii_types_found,
        "status":       status,
        "error":        _scrub(error) if error else None,
    })


def log_query(
    query_length: int,
    pii_in_query: bool,
    chunks_retrieved: int,
    chunks_relevant: int,
    status: str = "success",
) -> None:
    # Query text is NEVER logged — it may contain personal information
    _write({
        "event":                          "QUERY",
        "query_length_chars":             query_length,
        "pii_in_query":                   pii_in_query,
        "chunks_retrieved":               chunks_retrieved,
        "chunks_after_relevance_filter":  chunks_relevant,
        "status":                         status,
    })


def log_answer(
    answer_length: int,
    pii_in_answer: bool,
    grounded: bool,
    status: str = "success",
) -> None:
    # Answer text is NOT logged — may contain document content / PII
    _write({
        "event":               "ANSWER_GENERATED",
        "answer_length_chars": answer_length,
        "pii_in_answer":       pii_in_answer,
        "grounded":            grounded,
        "status":              status,
    })


def log_guardrail_violation(
    layer: str,
    violation_type: str,
    details: str,
) -> None:
    _write({
        "event":          "GUARDRAIL_VIOLATION",
        "layer":          layer,
        "violation_type": violation_type,
        "details":        _scrub(details),
    })


def log_guardrail_warning(
    layer: str,
    warning_type: str,
    details: str,
) -> None:
    _write({
        "event":        "GUARDRAIL_WARNING",
        "layer":        layer,
        "warning_type": warning_type,
        "details":      _scrub(details),
    })


def log_error(
    layer: str,
    error_type: str,
    details: str,
) -> None:
    _write({
        "event":      "SYSTEM_ERROR",
        "layer":      layer,
        "error_type": error_type,
        "details":    _scrub(details),
    })


def read_recent_logs(n: int = 50) -> List[Dict[str, Any]]:
    """Return the last n log entries from all rotated files as dicts."""
    entries: List[Dict[str, Any]] = []

    # Collect from rotated backups (oldest first) + current file
    log_files = []
    for i in range(BACKUP_COUNT, 0, -1):
        p = Path(f"{LOG_PATH}.{i}")
        if p.exists():
            log_files.append(p)
    log_files.append(Path(LOG_PATH))

    for log_file in log_files:
        if not log_file.exists():
            continue
        try:
            for line in log_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except OSError:
            continue

    # Return the most recent n entries, newest first
    return list(reversed(entries[-n:]))
