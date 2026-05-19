"""
Simple RAG System — Streamlit UI (fully local, no API key needed)
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

SUPPORTED = [".txt", ".md", ".pdf", ".docx", ".doc",
             ".xlsx", ".xls", ".csv", ".pptx"]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simple RAG",
    page_icon="🔍",
    layout="wide",
)

# ── shared resources (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)…")
def get_embedder():
    return Embedder()

@st.cache_resource
def get_store():
    return VectorStore()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Knowledge Base")
    st.caption("100% local · no API key needed")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[e.lstrip(".") for e in SUPPORTED],
        accept_multiple_files=True,
        help="PDF, DOCX, XLSX, CSV, PPTX, TXT, MD",
    )

    if st.button("Ingest uploaded files", type="primary",
                 disabled=not uploaded_files):
        embedder = get_embedder()
        store = get_store()
        total = 0

        for uf in uploaded_files:
            suffix = Path(uf.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            try:
                with st.spinner(f"Processing {uf.name}…"):
                    pages = load_file(tmp_path)
                    for p in pages:
                        p["source"] = uf.name

                    chunks, meta = [], []
                    for page in pages:
                        for i, c in enumerate(chunk_text(page["text"])):
                            chunks.append(c)
                            meta.append({
                                "source": uf.name,
                                "page": str(page["page"]),
                                "chunk_index": i,
                            })

                    embeddings = embedder.embed(chunks)
                    added = store.add(chunks, embeddings, meta)
                    total += added
                    st.success(f"✓ {uf.name} → {added} chunks")
            except Exception as e:
                st.error(f"✗ {uf.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if total:
            st.info(f"Total chunks in store: {store.count()}")
            st.rerun()

    st.divider()

    # Ingested documents list
    store = get_store()
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


# ── main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Simple RAG — Local")

# How it works banner
with st.expander("How this works", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**1. Ingest**\nUpload docs → split into chunks → embed with MiniLM")
    col2.info("**2. Retrieve**\nEmbed your question → cosine similarity → top-5 chunks")
    col3.info("**3. Generate**\nChunks + question → flan-t5-base → answer")
    col4.info("**4. Cite**\nEvery answer shows which doc + page it came from")

st.caption("Upload documents in the sidebar, then ask questions below.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved chunks", expanded=False):
                for i, h in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**{i}. {h['source']} — page {h['page']}** "
                        f"*(distance: {h['distance']})*"
                    )
                    st.code(h["text"], language=None)

# Chat input
if question := st.chat_input("Ask a question about your documents…"):
    store = get_store()

    if store.count() == 0:
        st.warning("Upload and ingest at least one document first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            embedder = get_embedder()
            query_vec = embedder.embed_one(question)
            hits = store.query(query_vec, n_results=5)

        with st.spinner("Generating answer with flan-t5-base…"):
            answer = generate_answer_local(question, hits)

        st.markdown(answer)

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
    })
