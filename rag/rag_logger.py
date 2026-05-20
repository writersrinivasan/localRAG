"""
RAG Inspector logger — per-query and per-ingest pipeline diagnostics.

Captures timing for each pipeline step (embed, retrieve, generate), embedding
vector statistics (dimension, L2 norm, mean, min, max), and per-chunk
similarity distances. Stored as JSONL in rag_diagnostics.log, separate from
the compliance audit trail in audit.log.

Log location: <project_root>/rag_diagnostics.log
Rotation: 5 MB × 5 backup files.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

LOG_PATH     = os.path.join(os.path.dirname(__file__), "..", "rag_diagnostics.log")
MAX_BYTES    = 5 * 1024 * 1024
BACKUP_COUNT = 5


class _RawLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("localrag.rag_inspector")
    if logger.handlers:
        return logger
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
        print(f"[RAG_LOGGER] Cannot open log file '{LOG_PATH}': {exc}", file=sys.stderr)
    return logger


_logger = _build_logger()


def _write(entry: Dict[str, Any]) -> None:
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        _logger.info(json.dumps(entry))
    except Exception as exc:
        print(f"[RAG_LOGGER WRITE ERROR] {exc} — entry: {entry}", file=sys.stderr)


# ── public logging functions ──────────────────────────────────────────────────

def log_ingest_diagnostics(
    filename: str,
    chunk_count: int,
    avg_chunk_size: float,
    min_chunk_size: int,
    max_chunk_size: int,
    embed_time_s: float,
    embed_dim: int,
) -> None:
    """Log embedding diagnostics captured during document ingest."""
    _write({
        "event":          "INGEST_DIAGNOSTICS",
        "filename":       filename,
        "chunk_count":    chunk_count,
        "avg_chunk_size": round(avg_chunk_size, 1),
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
        "embed_time_s":   round(embed_time_s, 3),
        "embed_dim":      embed_dim,
    })


def log_query_diagnostics(
    query_len: int,
    embed_time_s: float,
    embed_dim: int,
    embed_norm: float,
    embed_mean: float,
    embed_min: float,
    embed_max: float,
    retrieve_time_s: float,
    chunk_distances: List[float],
    context_chars: int,
    generate_time_s: float,
    answer_len: int,
    total_time_s: float,
) -> None:
    """Log full pipeline diagnostics for a single query."""
    _write({
        "event":           "QUERY_DIAGNOSTICS",
        "query_len":       query_len,
        "embed_time_s":    round(embed_time_s, 3),
        "embed_dim":       embed_dim,
        "embed_norm":      round(embed_norm, 4),
        "embed_mean":      round(embed_mean, 6),
        "embed_min":       round(embed_min, 6),
        "embed_max":       round(embed_max, 6),
        "retrieve_time_s": round(retrieve_time_s, 3),
        "chunk_distances": [round(d, 4) for d in chunk_distances],
        "context_chars":   context_chars,
        "generate_time_s": round(generate_time_s, 3),
        "answer_len":      answer_len,
        "total_time_s":    round(total_time_s, 3),
    })


def read_recent_logs(n: int = 100) -> List[Dict[str, Any]]:
    """Return the last n diagnostic entries from all rotated files, newest first."""
    entries: List[Dict[str, Any]] = []

    log_files: List[Path] = []
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

    return list(reversed(entries[-n:]))
