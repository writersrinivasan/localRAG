"""
Simple RAG System — Streamlit UI with guardrails and full error handling.
Run: streamlit run app.py
"""

import math
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import VectorStore
from rag.generator_local import generate_answer_local
from rag.exceptions import (
    RAGError, LoaderError, ChunkerError,
    EmbedderError, EmbedderTimeoutError,
    StoreError, StoreUnavailableError,
    GeneratorError, GeneratorTimeoutError,
)
from rag.guardrails import (
    file_guardrails, input_guardrails,
    retrieval_guardrails, output_guardrails, pii_detector,
)
from rag import audit_logger, rag_logger

SUPPORTED    = [".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx"]
MAX_MESSAGES = 50   # cap session history to prevent unbounded RAM growth

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="localRAG", page_icon="🔍", layout="wide")


# ── shared resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()


@st.cache_resource
def get_store():
    return VectorStore()


def _safe_get_embedder():
    """Return Embedder or show an error and stop the page."""
    try:
        return get_embedder()
    except EmbedderError as exc:
        st.error(f"**Embedding model failed to load:** {exc}")
        audit_logger.log_error("EMBEDDER", "MODEL_LOAD_FAILURE", str(exc))
        st.stop()


def _safe_get_store():
    """Return VectorStore or show an error and stop the page."""
    try:
        return get_store()
    except StoreUnavailableError as exc:
        st.error(f"**Vector store unavailable:** {exc}")
        audit_logger.log_error("STORE", "INIT_FAILURE", str(exc))
        st.stop()


# ── helpers ───────────────────────────────────────────────────────────────────
def show_result(result, *, location=st):
    for v in result.violations:
        location.error(f"**Blocked:** {v}")
    for w in result.warnings:
        location.warning(f"**Notice:** {w}")


def _safe_unlink(path: str) -> None:
    """Delete a file without raising if it is already gone."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _trim_messages() -> None:
    """Keep only the last MAX_MESSAGES entries to cap RAM usage."""
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]


def _vec_stats(vec):
    """Return (dim, norm, mean, min, max) for an embedding vector."""
    dim  = len(vec)
    norm = math.sqrt(sum(x * x for x in vec))
    mean = sum(vec) / dim
    return dim, norm, mean, min(vec), max(vec)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Knowledge Base")
    st.caption("100% local · no API key needed")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[e.lstrip(".") for e in SUPPORTED],
        accept_multiple_files=True,
        help="PDF, DOCX, XLSX, CSV, PPTX, TXT, MD — max 20 MB each",
    )

    if st.button("Ingest uploaded files", type="primary", disabled=not uploaded_files):
        embedder = _safe_get_embedder()
        store    = _safe_get_store()
        total    = 0

        for uf in uploaded_files:
            st.markdown(f"**{uf.name}**")
            file_bytes = uf.read()
            size_bytes = len(file_bytes)
            suffix     = Path(uf.name).suffix.lower()
            tmp_path   = None

            try:
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # ── GUARDRAIL 1: file validation ──────────────────────────
                file_result = file_guardrails.validate(tmp_path, uf.name, size_bytes)
                show_result(file_result)
                if not file_result.passed:
                    audit_logger.log_guardrail_violation(
                        "FILE_INGEST", "FILE_VALIDATION",
                        "; ".join(file_result.violations),
                    )
                    continue

                # ── parse ─────────────────────────────────────────────────
                with st.spinner("Extracting text…"):
                    pages = load_file(tmp_path)
                    for p in pages:
                        p["source"] = uf.name

                # ── GUARDRAIL 2: PII scan ─────────────────────────────────
                full_text  = " ".join(p["text"] for p in pages)
                pii_result = file_guardrails.scan_content_pii(full_text)
                pii_found  = pii_detector.scan(full_text)
                show_result(pii_result)
                if pii_result.warnings:
                    audit_logger.log_guardrail_warning(
                        "FILE_INGEST", "PII_IN_DOCUMENT",
                        f"{uf.name}: {', '.join(pii_found)}",
                    )

                # ── chunk ─────────────────────────────────────────────────
                chunks, meta = [], []
                for page in pages:
                    for i, c in enumerate(chunk_text(page["text"])):
                        chunks.append(c)
                        meta.append({
                            "source":      uf.name,
                            "page":        str(page["page"]),
                            "chunk_index": i,
                        })

                # ── embed + store ─────────────────────────────────────────
                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    _t_embed = time.perf_counter()
                    embeddings = embedder.embed(chunks)
                    _ingest_embed_time = time.perf_counter() - _t_embed

                added  = store.add(chunks, embeddings, meta)
                total += added

                # ── RAG Inspector: ingest diagnostics ─────────────────────
                _chunk_sizes = [len(c) for c in chunks]
                _ingest_dim  = len(embeddings[0]) if embeddings else 384
                rag_logger.log_ingest_diagnostics(
                    filename=uf.name,
                    chunk_count=len(chunks),
                    avg_chunk_size=sum(_chunk_sizes) / len(_chunk_sizes),
                    min_chunk_size=min(_chunk_sizes),
                    max_chunk_size=max(_chunk_sizes),
                    embed_time_s=_ingest_embed_time,
                    embed_dim=_ingest_dim,
                )

                audit_logger.log_ingest(
                    filename=uf.name, size_kb=size_bytes / 1024,
                    chunks_added=added, pii_types_found=pii_found,
                )
                st.success(f"✓ {added} chunks stored")

            except LoaderError as exc:
                st.error(f"**Cannot read file:** {exc}")
                audit_logger.log_ingest(
                    uf.name, size_bytes / 1024, 0, [],
                    status="error", error=str(exc),
                )

            except ChunkerError as exc:
                st.error(f"**Chunking failed:** {exc}")
                audit_logger.log_error("CHUNKER", "CHUNK_FAILURE", str(exc))

            except EmbedderTimeoutError as exc:
                st.error(f"**Embedding timed out:** {exc}")
                audit_logger.log_error("EMBEDDER", "TIMEOUT", str(exc))

            except EmbedderError as exc:
                st.error(f"**Embedding failed:** {exc}")
                audit_logger.log_error("EMBEDDER", "EMBED_FAILURE", str(exc))

            except StoreError as exc:
                st.error(f"**Storage failed:** {exc}")
                audit_logger.log_error("STORE", "WRITE_FAILURE", str(exc))

            except RAGError as exc:
                st.error(f"**Unexpected RAG error:** {exc}")
                audit_logger.log_error("INGEST", "RAG_ERROR", str(exc))

            except Exception as exc:
                st.error(f"**Unexpected error:** {exc}")
                audit_logger.log_error("INGEST", "UNKNOWN_ERROR", str(exc))

            finally:
                if tmp_path:
                    _safe_unlink(tmp_path)

        if total:
            st.info(f"Total chunks in store: {store.count()}")
            st.rerun()

    st.divider()

    try:
        store   = _safe_get_store()
        sources = store.list_sources()
    except StoreError as exc:
        st.error(f"Cannot read knowledge base: {exc}")
        sources = []

    if sources:
        st.markdown(f"**Ingested docs** ({store.count()} chunks)")
        for s in sources:
            st.markdown(f"- {s}")
        if st.button("Clear knowledge base", type="secondary"):
            try:
                store.clear()
                st.cache_resource.clear()
                st.rerun()
            except StoreError as exc:
                st.error(f"Clear failed: {exc}")
    else:
        st.caption("No documents ingested yet.")

    st.divider()
    st.caption("Embeddings: all-MiniLM-L6-v2\nGeneration: flan-t5-base\n(both run locally)")


# ── tabs ──────────────────────────────────────────────────────────────────────
chat_tab, rag_tab, audit_tab = st.tabs(["💬 Chat", "📊 RAG Inspector", "🛡️ Audit Log"])


# ── chat tab ──────────────────────────────────────────────────────────────────
with chat_tab:
    st.title("🔍 localRAG")

    with st.expander("How this works", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.info("**1. Ingest**\nDocs → chunks → embeddings → ChromaDB")
        c2.info("**2. Retrieve**\nEmbed query → cosine similarity → top-5 chunks")
        c3.info("**3. Generate**\nContext + question → flan-t5 → answer")
        c4.info("**4. Guardrails**\nFile · Input · Retrieval · Output · Audit")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for w in msg.get("warnings", []):
                st.warning(f"**Notice:** {w}")
            if msg.get("sources"):
                with st.expander("Retrieved chunks", expanded=False):
                    for i, h in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}. {h['source']} — page {h['page']}** "
                            f"*(distance: {h['distance']})*"
                        )
                        st.code(h["text"], language=None)

    if question := st.chat_input("Ask a question about your documents…"):
        store = _safe_get_store()

        # ── GUARDRAIL 3: input validation ──────────────────────────────────
        input_result = input_guardrails.validate(question)
        if not input_result.passed:
            for v in input_result.violations:
                st.error(f"**Blocked:** {v}")
            audit_logger.log_guardrail_violation(
                "QUERY_INPUT", "INPUT_VALIDATION",
                "; ".join(input_result.violations),
            )
            st.stop()

        with st.chat_message("user"):
            st.markdown(question)
            for w in input_result.warnings:
                st.warning(f"**Notice:** {w}")

        st.session_state.messages.append({
            "role":     "user",
            "content":  question,
            "warnings": input_result.warnings,
        })
        _trim_messages()

        with st.chat_message("assistant"):
            # ── GUARDRAIL 4: store not empty ───────────────────────────────
            try:
                count = store.count()
            except StoreError as exc:
                st.error(f"**Vector store error:** {exc}")
                audit_logger.log_error("STORE", "COUNT_FAILURE", str(exc))
                st.stop()

            if count == 0:
                st.warning("Upload and ingest at least one document first.")
                st.stop()

            # ── embed query + retrieve ──────────────────────────────────────
            try:
                with st.spinner("Searching knowledge base…"):
                    embedder = _safe_get_embedder()

                    _t0 = time.perf_counter()
                    query_vec = embedder.embed_one(question)
                    _embed_time = time.perf_counter() - _t0

                    _t1 = time.perf_counter()
                    raw_hits = store.query(query_vec, n_results=5)
                    _retrieve_time = time.perf_counter() - _t1

            except EmbedderTimeoutError as exc:
                st.error(f"**Embedding timed out:** {exc}")
                audit_logger.log_error("EMBEDDER", "TIMEOUT", str(exc))
                st.stop()
            except EmbedderError as exc:
                st.error(f"**Embedding failed:** {exc}")
                audit_logger.log_error("EMBEDDER", "EMBED_FAILURE", str(exc))
                st.stop()
            except StoreError as exc:
                st.error(f"**Search failed:** {exc}")
                audit_logger.log_error("STORE", "QUERY_FAILURE", str(exc))
                st.stop()

            # ── GUARDRAIL 5: relevance threshold ───────────────────────────
            retrieval_result = retrieval_guardrails.validate(raw_hits)
            if not retrieval_result.passed:
                show_result(retrieval_result)
                audit_logger.log_guardrail_violation(
                    "RETRIEVAL", "LOW_RELEVANCE",
                    "; ".join(retrieval_result.violations),
                )
                st.stop()

            hits = retrieval_guardrails.filter_relevant(raw_hits)
            audit_logger.log_query(
                query_length=len(question),
                pii_in_query=bool(input_result.warnings),
                chunks_retrieved=len(raw_hits),
                chunks_relevant=len(hits),
            )

            # ── generate ───────────────────────────────────────────────────
            try:
                with st.spinner("Generating answer…"):
                    _t2 = time.perf_counter()
                    answer = generate_answer_local(question, hits)
                    _generate_time = time.perf_counter() - _t2

            except GeneratorTimeoutError as exc:
                st.error(f"**Generation timed out:** {exc}")
                audit_logger.log_error("GENERATOR", "TIMEOUT", str(exc))
                st.stop()
            except GeneratorError as exc:
                st.error(f"**Generation failed:** {exc}")
                audit_logger.log_error("GENERATOR", "GENERATE_FAILURE", str(exc))
                st.stop()

            # ── GUARDRAIL 6: output validation ─────────────────────────────
            output_result   = output_guardrails.validate(answer, hits)
            output_warnings = output_result.warnings

            audit_logger.log_answer(
                answer_length=len(answer),
                pii_in_answer=any("PII" in w for w in output_warnings),
                grounded=not any("grounded" in w.lower() for w in output_warnings),
            )
            if output_warnings:
                audit_logger.log_guardrail_warning(
                    "OUTPUT", "OUTPUT_QUALITY",
                    "; ".join(output_warnings),
                )

            # ── RAG Inspector: query diagnostics ───────────────────────────
            _total_time = _embed_time + _retrieve_time + _generate_time
            _dim, _norm, _mean, _vmin, _vmax = _vec_stats(query_vec)
            rag_logger.log_query_diagnostics(
                query_len=len(question),
                embed_time_s=_embed_time,
                embed_dim=_dim,
                embed_norm=_norm,
                embed_mean=_mean,
                embed_min=_vmin,
                embed_max=_vmax,
                retrieve_time_s=_retrieve_time,
                chunk_distances=[h["distance"] for h in hits],
                context_chars=sum(len(h["text"]) for h in hits),
                generate_time_s=_generate_time,
                answer_len=len(answer),
                total_time_s=_total_time,
            )

            st.markdown(answer)
            for w in output_warnings:
                st.warning(f"**Notice:** {w}")

            with st.expander("Retrieved chunks", expanded=False):
                for i, h in enumerate(hits, 1):
                    st.markdown(
                        f"**{i}. {h['source']} — page {h['page']}** "
                        f"*(distance: {h['distance']})*"
                    )
                    st.code(h["text"], language=None)

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  answer,
            "sources":  hits,
            "warnings": output_warnings,
        })
        _trim_messages()


# ── RAG Inspector tab ─────────────────────────────────────────────────────────
with rag_tab:
    st.title("📊 RAG Inspector")
    st.caption(
        "Live pipeline diagnostics: per-step timing, embedding vector statistics, "
        "and chunk similarity scores. Logged to rag_diagnostics.log."
    )

    if st.button("Refresh", key="rag_refresh"):
        st.rerun()

    try:
        diag_logs   = rag_logger.read_recent_logs(n=200)
    except Exception as exc:
        st.error(f"Cannot read RAG diagnostics log: {exc}")
        diag_logs = []

    query_logs  = [l for l in diag_logs if l.get("event") == "QUERY_DIAGNOSTICS"]
    ingest_logs = [l for l in diag_logs if l.get("event") == "INGEST_DIAGNOSTICS"]

    # ── summary metrics ────────────────────────────────────────────────────
    def _avg(logs, key):
        vals = [l[key] for l in logs if key in l]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total Queries",    len(query_logs))
    mc2.metric("Total Ingests",    len(ingest_logs))
    mc3.metric("Avg Embed (s)",    _avg(query_logs, "embed_time_s"))
    mc4.metric("Avg Retrieve (s)", _avg(query_logs, "retrieve_time_s"))
    mc5.metric("Avg Generate (s)", _avg(query_logs, "generate_time_s"))

    st.divider()

    # ── last query breakdown ───────────────────────────────────────────────
    if query_logs:
        last = query_logs[0]   # newest first
        ts   = last.get("timestamp", "")[:19].replace("T", " ")
        st.subheader(f"Last Query  —  `{ts} UTC`")

        # Pipeline timing visualization
        st.markdown("**Pipeline Timing**")
        steps = {
            "🔢 Embed query": last.get("embed_time_s", 0),
            "🔍 Retrieve":    last.get("retrieve_time_s", 0),
            "✍️ Generate":    last.get("generate_time_s", 0),
        }
        total_t = last.get("total_time_s", 1) or 1
        for label, t in steps.items():
            col_label, col_bar, col_val = st.columns([2, 6, 2])
            col_label.markdown(label)
            col_bar.progress(min(t / total_t, 1.0))
            col_val.markdown(f"`{t:.3f} s`")
        st.caption(f"Total pipeline time: **{last.get('total_time_s', 0):.3f} s**")

        st.divider()

        # Embedding vector statistics
        st.markdown("**Embedding Vector Statistics**")
        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        vc1.metric("Dimensions",  last.get("embed_dim", 384))
        vc2.metric("L2 Norm",     f"{last.get('embed_norm', 0):.3f}")
        vc3.metric("Mean",        f"{last.get('embed_mean', 0):.4f}")
        vc4.metric("Min",         f"{last.get('embed_min', 0):.4f}")
        vc5.metric("Max",         f"{last.get('embed_max', 0):.4f}")
        st.caption(
            "The L2 norm of a unit-normalised sentence-transformer vector is typically close to 1. "
            "Large norms indicate un-normalised vectors from a custom model."
        )

        st.divider()

        # Chunk similarity scores
        distances = last.get("chunk_distances", [])
        if distances:
            st.markdown("**Retrieved Chunk Similarity** — lower distance = more relevant")
            for i, dist in enumerate(distances, 1):
                similarity = max(0.0, 1.0 - dist)   # cosine distance → similarity
                col_rank, col_bar, col_score = st.columns([1, 7, 2])
                col_rank.markdown(f"Chunk {i}")
                col_bar.progress(similarity)
                col_score.markdown(f"`dist: {dist:.4f}`")

            st.caption(
                f"Query: **{last.get('query_len', 0)} chars** · "
                f"Context: **{last.get('context_chars', 0)} chars** · "
                f"Answer: **{last.get('answer_len', 0)} chars**"
            )

    else:
        st.info("No query diagnostics yet. Ask a question in the Chat tab first.")

    st.divider()

    # ── query history table ────────────────────────────────────────────────
    if query_logs:
        st.subheader("Query History")
        rows = []
        for entry in query_logs:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            rows.append({
                "Time (UTC)":   ts,
                "Query len":    entry.get("query_len", ""),
                "Embed (s)":    entry.get("embed_time_s", ""),
                "Retrieve (s)": entry.get("retrieve_time_s", ""),
                "Generate (s)": entry.get("generate_time_s", ""),
                "Total (s)":    entry.get("total_time_s", ""),
                "Chunks":       len(entry.get("chunk_distances", [])),
                "Answer len":   entry.get("answer_len", ""),
                "Embed dim":    entry.get("embed_dim", ""),
                "L2 norm":      entry.get("embed_norm", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── ingest history table ───────────────────────────────────────────────
    if ingest_logs:
        st.subheader("Ingest History")
        rows = []
        for entry in ingest_logs:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            rows.append({
                "Time (UTC)":     ts,
                "File":           entry.get("filename", ""),
                "Chunks":         entry.get("chunk_count", ""),
                "Avg chunk size": entry.get("avg_chunk_size", ""),
                "Min chunk size": entry.get("min_chunk_size", ""),
                "Max chunk size": entry.get("max_chunk_size", ""),
                "Embed time (s)": entry.get("embed_time_s", ""),
                "Embed dim":      entry.get("embed_dim", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif not query_logs:
        st.info("No ingest diagnostics yet. Upload and ingest a document first.")


# ── audit log tab ─────────────────────────────────────────────────────────────
with audit_tab:
    st.title("🛡️ Audit Log")
    st.caption(
        "Rotating compliance trail (5 MB × 5 files → audit.log). "
        "Query text and document content are never logged."
    )

    if st.button("Refresh"):
        st.rerun()

    try:
        logs = audit_logger.read_recent_logs(n=100)
    except Exception as exc:
        st.error(f"Cannot read audit log: {exc}")
        logs = []

    if not logs:
        st.info("No audit events yet. Ingest a document or ask a question.")
    else:
        events = [l["event"] for l in logs]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Events",   len(logs))
        c2.metric("Files Ingested", events.count("FILE_INGESTED"))
        c3.metric("Queries",        events.count("QUERY"))
        c4.metric("Violations",     events.count("GUARDRAIL_VIOLATION"))
        c5.metric("System Errors",  events.count("SYSTEM_ERROR"))

        st.divider()

        for entry in logs:
            evt = entry.get("event", "")
            ts  = entry.get("timestamp", "")[:19].replace("T", " ")

            if evt in ("GUARDRAIL_VIOLATION", "SYSTEM_ERROR") or entry.get("status") == "error":
                badge = "🔴"
            elif evt == "GUARDRAIL_WARNING":
                badge = "🟡"
            else:
                badge = "🟢"

            details = {k: v for k, v in entry.items() if k not in ("event", "timestamp")}
            with st.expander(f"{badge} `{ts}` — **{evt}**"):
                st.json(details)
