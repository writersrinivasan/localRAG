# Memory Management in localRAG

This document explains exactly what lives in memory, where it lives, how long it stays, and what the size implications are — across every layer of the app.

---

## The Five Memory Zones

```
┌─────────────────────────────────────────────────────────────────┐
│  ZONE 1: Process RAM (models)          ~400 MB, lives forever   │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  all-MiniLM-L6-v2   │  │      flan-t5-base pipeline      │  │
│  │  ~90 MB              │  │      ~300 MB                    │  │
│  │  @st.cache_resource  │  │      @st.cache_resource         │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ZONE 2: Session RAM (chat history)    ~KB per message          │
│  st.session_state["messages"]  →  grows with every Q&A pair    │
├─────────────────────────────────────────────────────────────────┤
│  ZONE 3: Ephemeral RAM (per-request)   freed after each action  │
│  file bytes → chunks → embeddings → query vector → hits        │
├─────────────────────────────────────────────────────────────────┤
│  ZONE 4: Disk — ChromaDB              grows with each ingest    │
│  chroma_db/   →  HNSW index + SQLite + raw chunk text          │
├─────────────────────────────────────────────────────────────────┤
│  ZONE 5: Disk — Caches & Logs         one-time + append-only   │
│  ~/.cache/huggingface/   audit.log                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Zone 1 — Model RAM (Permanent for process lifetime)

Both models are loaded with `@st.cache_resource`, which means they are:
- **Created once** when first needed
- **Shared across all browser sessions** (not per-user)
- **Never garbage collected** while the Streamlit server is running
- **Cleared only** by calling `st.cache_resource.clear()` or restarting the server

### Embedding model — `all-MiniLM-L6-v2`

| Property | Detail |
|---|---|
| Loaded in | `rag/embedder.py` → `Embedder.__init__()` |
| Cached by | `@st.cache_resource` on `get_embedder()` in `app.py` |
| RAM footprint | ~90 MB |
| When loaded | First page load that triggers `get_embedder()` |
| Used for | Converting chunks to vectors (ingest) and query to vector (query time) |
| Freed when | Streamlit server restarts or `st.cache_resource.clear()` |

### Generation model — `google/flan-t5-base`

| Property | Detail |
|---|---|
| Loaded in | `rag/generator_local.py` → `_get_pipeline()` |
| Cached by | `@st.cache_resource` on `_get_pipeline()` |
| RAM footprint | ~300 MB (encoder + decoder weights + tokenizer) |
| When loaded | First query submitted after ingest |
| Used for | Generating answers from retrieved context |
| Freed when | Streamlit server restarts or `st.cache_resource.clear()` |

**Total model RAM: ~390 MB** — held permanently for the lifetime of the process.

---

## Zone 2 — Session RAM (Per browser tab)

### Chat history — `st.session_state["messages"]`

Each message stored in session state is a dict:

```python
# User message
{"role": "user", "content": "<question text>", "warnings": [...]}

# Assistant message
{
    "role": "assistant",
    "content": "<answer text>",       # up to 200 tokens (~200 chars)
    "sources": [                       # up to 5 chunks
        {"text": "...", "source": "file.pdf", "page": "3", "distance": 0.42},
        ...
    ],
    "warnings": [...]
}
```

**Size per exchange:**
- Question: ~100–500 chars
- Answer: ~200–800 chars
- 5 source chunks × ~500 chars each = ~2,500 chars
- ~3–4 KB per Q&A pair

**Key behaviour:** This list grows indefinitely within a browser session. There is no automatic cleanup or size cap. Closing the browser tab destroys it; refreshing the page resets it.

---

## Zone 3 — Ephemeral RAM (Per-request, freed immediately after use)

These objects exist only during a single ingest or query cycle and are released by Python's garbage collector as soon as the operation completes.

### During ingest

```
uf.read()                          # raw file bytes — max 20 MB (guardrail limit)
    │
    └─▶ tempfile on disk           # written immediately, deleted in finally block
         │
         └─▶ pages[]              # extracted text per page/sheet
              │
              └─▶ chunks[]        # list of ~500-char strings
                   │
                   └─▶ embeddings[]   # list of 384-float lists
                        │              # 384 × 4 bytes × N chunks
                        └─▶ store.add() ──▶ ChromaDB (Zone 4)
                             │
                             └─▶ all of the above freed ✓
```

**Temporary disk:** `tempfile.NamedTemporaryFile` is created per file, deleted in the `finally` block of the ingest loop regardless of success or failure. The 20 MB guardrail caps how large this can be.

**Embedding list size:** For 100 chunks: `100 × 384 × 4 bytes ≈ 150 KB`. Negligible.

### During query

```
question (string)
    │
    └─▶ embed_one() → query_vec   # 384 floats = 1.5 KB
         │
         └─▶ store.query()        # ChromaDB reads from disk into RAM
              │
              └─▶ hits[]          # 5 dicts, ~3 KB total
                   │
                   ├─▶ retrieval_guardrails.filter_relevant(hits)
                   ├─▶ generate_answer_local(question, hits)  # prompt string ~2 KB
                   │
                   └─▶ answer (string, ~200–800 chars)
                        │
                        └─▶ appended to st.session_state["messages"] (Zone 2)
                             │
                             └─▶ all intermediates freed ✓
```

---

## Zone 4 — ChromaDB (Disk, persistent)

ChromaDB stores everything in `chroma_db/` at the project root. This directory survives process restarts.

### What's stored

| File | Contents |
|---|---|
| `chroma_db/chroma.sqlite3` | Collection metadata, document text, chunk metadata (source, page, chunk_index) |
| `chroma_db/<uuid>/index/` | HNSW index files — the actual embedding vectors |
| `chroma_db/<uuid>/header.bin` | Index configuration |

### Size estimate per chunk

| Component | Size |
|---|---|
| Raw chunk text (~500 chars) | ~500 B |
| Embedding vector (384 × float32) | 1,536 B |
| Metadata (source, page, chunk_index) | ~100 B |
| **Per chunk total** | **~2.1 KB** |

**Example:** A 20-page PDF produces roughly 80–120 chunks → ~200–250 KB in ChromaDB.

### How ChromaDB reads data into RAM

ChromaDB does **not** load the entire index into RAM on startup. The HNSW index is **memory-mapped** — only the portions needed to answer a specific query are pulled into RAM, and the OS page cache manages eviction. For a typical corpus of a few hundred chunks, the full index fits comfortably in RAM and stays there passively.

### Clearing

```python
store.clear()   # deletes the collection and recreates it empty
                # chroma_db/ files are removed; disk space is reclaimed
```

---

## Zone 5 — Disk Caches and Audit Log

### HuggingFace model cache

| Model | Cache location | Size |
|---|---|---|
| `all-MiniLM-L6-v2` | `~/.cache/huggingface/hub/` | ~90 MB |
| `google/flan-t5-base` | `~/.cache/huggingface/hub/` | ~250 MB |

Downloaded once on first use. Never deleted automatically. To free:
```bash
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-base
```

### Audit log — `audit.log`

| Property | Detail |
|---|---|
| Format | JSONL (one JSON object per line) |
| Location | `<project_root>/audit.log` |
| Size per entry | ~200–400 bytes |
| Growth rate | One entry per ingest file, one per query, one per answer, one per violation |
| Rotation | **None currently** — grows indefinitely |
| Cleanup | Manual: `> audit.log` to truncate (preserves file), or `rm audit.log` |

---

## Full Memory Lifecycle

```
SERVER START
│
├── Python imports (no models in RAM yet)
│
▼
FIRST PAGE LOAD
│
├── get_embedder() hit → load all-MiniLM-L6-v2 → ~90 MB locked in RAM
│
▼
INGEST A FILE
│
├── File bytes → RAM (≤ 20 MB, guardrail-enforced)
├── Written to tempfile → disk
├── Text extracted → chunk list → embed list → ChromaDB write → disk
├── All ingest intermediates freed
└── tempfile deleted (finally block)
│
▼
FIRST QUERY
│
├── _get_pipeline() hit → load flan-t5-base → ~300 MB locked in RAM
├── Query embedded → 1.5 KB vector → ChromaDB cosine search
├── 5 chunks loaded from disk → ~3 KB in RAM
├── Prompt built → answer generated → answer string in RAM
├── answer + chunks appended to session_state["messages"]
└── intermediates (vector, prompt string) freed
│
▼
SUBSEQUENT QUERIES
│
├── Models already in RAM (no reload)
├── session_state["messages"] grows by ~3-4 KB per exchange
└── ChromaDB reads remain disk-backed (memory-mapped)
│
▼
BROWSER TAB CLOSED
│
└── session_state["messages"] destroyed — Zone 2 RAM reclaimed

SERVER RESTART
│
├── Zone 1 (model RAM) fully released
├── Zone 2 (session state) fully released
├── Zone 3 (ephemeral) fully released
├── Zone 4 (ChromaDB on disk) SURVIVES — documents still queryable
└── Zone 5 (audit.log, HF cache) SURVIVES
```

---

## What This App Does NOT Have (and Why It Matters)

| Missing | Risk | Mitigation if needed |
|---|---|---|
| Session state size cap | Long conversations accumulate MB of chunk text in RAM | Add `MAX_MESSAGES = 20` and slice `st.session_state.messages[-MAX_MESSAGES:]` |
| Audit log rotation | `audit.log` grows indefinitely | Add `logging.handlers.RotatingFileHandler` with a size cap |
| Model unloading on idle | ~390 MB held even if no one is using the app | Use `@st.cache_resource(ttl=3600)` to auto-expire after 1 hour of inactivity |
| Multi-user isolation | All users share the same ChromaDB collection | Add a per-user collection prefix if multi-tenancy is required |
| ChromaDB size limit | `chroma_db/` grows without bound | Add a pre-ingest check: `if store.count() > MAX_CHUNKS: block` |
