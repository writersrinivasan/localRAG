# localRAG — Simple Local RAG System

A minimal, fully local Retrieval-Augmented Generation (RAG) system built for learning the basics. No API keys, no cloud services — everything runs on your machine.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that grounds an LLM's answers in your own documents:

```
Your Documents
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  1. Ingest  │────▶│  2. Retrieve │────▶│  3. Generate    │
│             │     │              │     │                 │
│ Load → Chunk│     │ Embed query  │     │ Context +       │
│ → Embed     │     │ → Find top-5 │     │ Question →      │
│ → Store     │     │   similar    │     │ flan-t5 → Answer│
└─────────────┘     └──────────────┘     └─────────────────┘
```

Instead of relying on the model's training data, it first **retrieves** relevant passages from your documents and then **generates** an answer grounded in that context.

---

## Tech Stack

| Role | Tool | Why |
|---|---|---|
| Document parsing | pypdf, python-docx, pandas, python-pptx | Cover all common formats |
| Text chunking | Custom sliding-window | Simple, no dependencies |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, 384-dim, local |
| Vector store | ChromaDB (persisted to disk) | Simple, no server needed |
| Generation | `google/flan-t5-base` (HuggingFace) | Free, local, no API key |
| UI | Streamlit | Quick interactive web UI |

---

## Supported File Types

| Format | Extension |
|---|---|
| Plain text / Markdown | `.txt`, `.md` |
| PDF | `.pdf` |
| Word document | `.docx`, `.doc` |
| Excel spreadsheet | `.xlsx`, `.xls` |
| CSV | `.csv` |
| PowerPoint | `.pptx` |

---

## Project Structure

```
localRAG/
│
├── app.py                  # Streamlit web UI (recommended)
├── main.py                 # CLI alternative
├── requirements.txt
├── .env.example
│
├── docs/                   # Drop your documents here (for CLI use)
│
└── rag/
    ├── loader.py           # Parse PDF, DOCX, Excel, CSV, PPTX, TXT → text
    ├── chunker.py          # Sliding-window text splitter (500 chars, 100 overlap)
    ├── embedder.py         # Sentence-transformers wrapper (all-MiniLM-L6-v2)
    ├── store.py            # ChromaDB wrapper — add / query / list / clear
    ├── generator_local.py  # Local generation via flan-t5-base (no API key)
    └── generator.py        # Optional: Claude API generation (requires key)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/writersrinivasan/localRAG.git
cd localRAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned because PyTorch and sentence-transformers require NumPy 1.x.

### 3. (Optional) API key for Claude generation

Only needed if you want to use `generator.py` instead of the local model.

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Web UI (recommended)

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Steps:**
1. Upload one or more documents using the sidebar uploader
2. Click **"Ingest uploaded files"**
3. Type a question in the chat box at the bottom
4. The answer appears with source citations and a "Retrieved chunks" expander

> On first use, two models download automatically:
> - `all-MiniLM-L6-v2` (~90 MB) — for embeddings
> - `google/flan-t5-base` (~250 MB) — for answer generation
>
> Both are cached after the first download.

---

### CLI

```bash
# Add a single file
python main.py ingest path/to/document.pdf

# Add an entire folder
python main.py ingest docs/

# Ask a question
python main.py query "What are the main conclusions?"

# See what's been ingested
python main.py list

# Wipe the knowledge base
python main.py clear
```

---

## How Each File Works

### `rag/loader.py`
Reads a file and returns a list of `{text, source, page}` dicts. Each format has its own private function (`_load_pdf`, `_load_docx`, etc.) that the public `load_file()` dispatches to based on file extension.

### `rag/chunker.py`
Splits long text into overlapping windows. The **overlap** (100 chars by default) ensures that a sentence split across two chunk boundaries doesn't lose context — the tail of one chunk repeats at the start of the next.

### `rag/embedder.py`
Wraps `SentenceTransformer("all-MiniLM-L6-v2")`. Converts a list of strings into 384-dimensional float vectors. The same model is used for both document chunks (at ingest time) and the user query (at query time) — this is what makes cosine similarity meaningful.

### `rag/store.py`
Thin wrapper around ChromaDB. Persists to `./chroma_db/` on disk so ingested documents survive between runs. Uses **cosine similarity** as the distance metric.

### `rag/generator_local.py`
Uses HuggingFace's `text2text-generation` pipeline with `google/flan-t5-base`. Builds a prompt of the form:
```
Answer the question based only on the context below.
Context: <retrieved chunks>
Question: <user question>
Answer:
```
Context is truncated to ~1800 characters to stay within flan-t5's 512-token input limit.

### `rag/generator.py`
Alternative generator that calls the Claude API. Requires `ANTHROPIC_API_KEY` in `.env`.

---

## Key Concepts Illustrated

| Concept | Where to look |
|---|---|
| Document ingestion pipeline | `loader.py` → `chunker.py` → `embedder.py` → `store.py` |
| Semantic search (not keyword) | `store.py` — cosine similarity on embeddings |
| Chunk overlap and why it matters | `chunker.py` — `overlap` parameter |
| Grounded generation | `generator_local.py` — context injected into prompt |
| Source citation | `app.py` — "Retrieved chunks" expander shows exact passages |

---

## License

MIT
