"""
Simple RAG System — Streamlit UI with guardrails at every layer.
Run: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import VectorStore
from rag.generator_local import generate_answer_local
from rag.guardrails import (
    file_guardrails,
    input_guardrails,
    retrieval_guardrails,
    output_guardrails,
    pii_detector,
)
from rag import audit_logger

SUPPORTED = [".txt", ".md", ".pdf", ".docx", ".doc",
             ".xlsx", ".xls", ".csv", ".pptx"]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="localRAG",
    page_icon="🔍",
    layout="wide",
)

# ── shared resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()

@st.cache_resource
def get_store():
    return VectorStore()


# ── helpers ───────────────────────────────────────────────────────────────────
def show_result(result, *, location=st):
    """Render guardrail violations (red) and warnings (yellow)."""
    for v in result.violations:
        location.error(f"**Blocked:** {v}")
    for w in result.warnings:
        location.warning(f"**Notice:** {w}")


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

    if st.button("Ingest uploaded files", type="primary",
                 disabled=not uploaded_files):

        embedder = get_embedder()
        store    = get_store()
        total    = 0

        for uf in uploaded_files:
            st.markdown(f"**{uf.name}**")
            file_bytes  = uf.read()
            size_bytes  = len(file_bytes)
            suffix      = Path(uf.name).suffix.lower()

            # ── GUARDRAIL 1: file validation ──────────────────────────────
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            file_result = file_guardrails.validate(tmp_path, uf.name, size_bytes)
            show_result(file_result)

            if not file_result.passed:
                audit_logger.log_guardrail_violation(
                    layer="FILE_INGEST",
                    violation_type="FILE_VALIDATION",
                    details="; ".join(file_result.violations),
                )
                os.unlink(tmp_path)
                continue

            try:
                with st.spinner(f"Extracting text…"):
                    pages = load_file(tmp_path)
                    for p in pages:
                        p["source"] = uf.name

                # ── GUARDRAIL 2: PII scan on ingested content ─────────────
                full_text    = " ".join(p["text"] for p in pages)
                pii_result   = file_guardrails.scan_content_pii(full_text)
                pii_found    = pii_detector.scan(full_text)
                show_result(pii_result)

                if pii_result.warnings:
                    audit_logger.log_guardrail_warning(
                        layer="FILE_INGEST",
                        warning_type="PII_IN_DOCUMENT",
                        details=f"{uf.name}: {', '.join(pii_found)}",
                    )

                chunks, meta = [], []
                for page in pages:
                    for i, c in enumerate(chunk_text(page["text"])):
                        chunks.append(c)
                        meta.append({
                            "source": uf.name,
                            "page": str(page["page"]),
                            "chunk_index": i,
                        })

                with st.spinner("Embedding chunks…"):
                    embeddings = embedder.embed(chunks)
                    added      = store.add(chunks, embeddings, meta)
                    total     += added

                audit_logger.log_ingest(
                    filename=uf.name,
                    size_kb=size_bytes / 1024,
                    chunks_added=added,
                    pii_types_found=pii_found,
                )
                st.success(f"✓ {added} chunks stored")

            except Exception as e:
                audit_logger.log_ingest(
                    filename=uf.name, size_kb=size_bytes / 1024,
                    chunks_added=0, pii_types_found=[],
                    status="error", error=str(e),
                )
                st.error(f"Failed to process: {e}")
            finally:
                os.unlink(tmp_path)

        if total:
            st.info(f"Total chunks in store: {store.count()}")
            st.rerun()

    st.divider()

    store   = get_store()
    sources = store.list_sources()
    if sources:
        st.markdown(f"**Ingested docs** ({store.count()} chunks)")
        for s in sources:
            st.markdown(f"- {s}")
        if st.button("Clear knowledge base", type="secondary"):
            store.clear()
            st.cache_resource.clear()
            st.rerun()
    else:
        st.caption("No documents ingested yet.")

    st.divider()
    st.caption("Embeddings: all-MiniLM-L6-v2\nGeneration: flan-t5-base\n(both run locally)")


# ── main tabs ─────────────────────────────────────────────────────────────────
chat_tab, audit_tab = st.tabs(["💬 Chat", "🛡️ Audit Log"])


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
        store = get_store()

        # ── GUARDRAIL 3: input validation ─────────────────────────────────
        input_result = input_guardrails.validate(question)

        if not input_result.passed:
            for v in input_result.violations:
                st.error(f"**Blocked:** {v}")
            audit_logger.log_guardrail_violation(
                layer="QUERY_INPUT",
                violation_type="INPUT_VALIDATION",
                details="; ".join(input_result.violations),
            )
            st.stop()

        with st.chat_message("user"):
            st.markdown(question)
            for w in input_result.warnings:
                st.warning(f"**Notice:** {w}")

        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "warnings": input_result.warnings,
        })

        with st.chat_message("assistant"):
            # ── GUARDRAIL 4: store not empty ──────────────────────────────
            if store.count() == 0:
                st.warning("Upload and ingest at least one document first.")
                st.stop()

            with st.spinner("Searching knowledge base…"):
                embedder   = get_embedder()
                query_vec  = embedder.embed_one(question)
                raw_hits   = store.query(query_vec, n_results=5)

            # ── GUARDRAIL 5: relevance threshold ──────────────────────────
            retrieval_result = retrieval_guardrails.validate(raw_hits)
            if not retrieval_result.passed:
                show_result(retrieval_result)
                audit_logger.log_guardrail_violation(
                    layer="RETRIEVAL",
                    violation_type="LOW_RELEVANCE",
                    details="; ".join(retrieval_result.violations),
                )
                st.stop()

            hits = retrieval_guardrails.filter_relevant(raw_hits)

            audit_logger.log_query(
                query_length=len(question),
                pii_in_query=bool(input_result.warnings),
                chunks_retrieved=len(raw_hits),
                chunks_relevant=len(hits),
            )

            with st.spinner("Generating answer…"):
                answer = generate_answer_local(question, hits)

            # ── GUARDRAIL 6: output validation ────────────────────────────
            output_result = output_guardrails.validate(answer, hits)
            output_warnings = output_result.warnings

            audit_logger.log_answer(
                answer_length=len(answer),
                pii_in_answer=any("PII" in w for w in output_warnings),
                grounded=not any("grounded" in w.lower() for w in output_warnings),
            )

            if output_warnings:
                audit_logger.log_guardrail_warning(
                    layer="OUTPUT",
                    warning_type="OUTPUT_QUALITY",
                    details="; ".join(output_warnings),
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
            "role": "assistant",
            "content": answer,
            "sources": hits,
            "warnings": output_warnings,
        })


# ── audit log tab ─────────────────────────────────────────────────────────────
with audit_tab:
    st.title("🛡️ Audit Log")
    st.caption("Append-only compliance trail — written to audit.log. "
               "Query text and document content are never logged.")

    if st.button("Refresh"):
        st.rerun()

    logs = audit_logger.read_recent_logs(n=100)

    if not logs:
        st.info("No audit events yet. Ingest a document or ask a question.")
    else:
        # Summary metrics
        events = [l["event"] for l in logs]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Events",      len(logs))
        col2.metric("Files Ingested",    events.count("FILE_INGESTED"))
        col3.metric("Queries",           events.count("QUERY"))
        col4.metric("Violations",        events.count("GUARDRAIL_VIOLATION"))

        st.divider()

        # Color-coded log entries
        for entry in logs:
            evt = entry.get("event", "")
            ts  = entry.get("timestamp", "")[:19].replace("T", " ")

            if evt == "GUARDRAIL_VIOLATION":
                badge = "🔴"
                color = "red"
            elif evt == "GUARDRAIL_WARNING":
                badge = "🟡"
                color = "orange"
            elif evt == "FILE_INGESTED" and entry.get("status") == "error":
                badge = "🔴"
                color = "red"
            else:
                badge = "🟢"
                color = "green"

            details = {k: v for k, v in entry.items()
                       if k not in ("event", "timestamp")}

            with st.expander(f"{badge} `{ts}` — **{evt}**"):
                st.json(details)
