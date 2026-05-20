"""
Audit logger — append-only JSONL audit trail for compliance.

Every ingest, query, generation, and guardrail event is logged to
audit.log as a JSON line. Each entry has a timestamp, event type,
and structured details. PII is never written to the log.

Log location: <project_root>/audit.log
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "audit.log")

# Rough PII scrub for log values — belt-and-suspenders before writing
_PII_SCRUB = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"   # email
    r"|\b\d{3}-\d{2}-\d{4}\b"                                    # SSN
    r"|\b(?:\d{4}[\s\-]?){3}\d{4}\b"                            # credit card
)


def _scrub(value: str) -> str:
    return _PII_SCRUB.sub("[REDACTED]", str(value))


def _write(entry: Dict[str, Any]) -> None:
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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
        "event": "FILE_INGESTED",
        "file": filename,
        "size_kb": round(size_kb, 1),
        "chunks_added": chunks_added,
        "pii_detected": bool(pii_types_found),
        "pii_types": pii_types_found,
        "status": status,
        "error": error,
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
        "event": "QUERY",
        "query_length_chars": query_length,
        "pii_in_query": pii_in_query,
        "chunks_retrieved": chunks_retrieved,
        "chunks_after_relevance_filter": chunks_relevant,
        "status": status,
    })


def log_answer(
    answer_length: int,
    pii_in_answer: bool,
    grounded: bool,
    status: str = "success",
) -> None:
    # Answer text is NOT logged — may contain document content / PII
    _write({
        "event": "ANSWER_GENERATED",
        "answer_length_chars": answer_length,
        "pii_in_answer": pii_in_answer,
        "grounded": grounded,
        "status": status,
    })


def log_guardrail_violation(
    layer: str,
    violation_type: str,
    details: str,
) -> None:
    _write({
        "event": "GUARDRAIL_VIOLATION",
        "layer": layer,
        "violation_type": violation_type,
        "details": _scrub(details),
    })


def log_guardrail_warning(
    layer: str,
    warning_type: str,
    details: str,
) -> None:
    _write({
        "event": "GUARDRAIL_WARNING",
        "layer": layer,
        "warning_type": warning_type,
        "details": _scrub(details),
    })


def read_recent_logs(n: int = 50) -> List[Dict[str, Any]]:
    """Return the last n log entries as dicts."""
    path = Path(LOG_PATH)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return list(reversed(entries))
