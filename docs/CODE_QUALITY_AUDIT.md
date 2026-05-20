# Code Quality Audit — localRAG

Independent audit across 10 dimensions. Every finding cites the exact file and line.

---

## Overall Score: 7.3 / 10

| Dimension | Score |
|---|---|
| Code Structure & Modularity | 8.0 |
| Readability & Naming | 8.5 |
| Error Handling & Resilience | 6.5 |
| Security & Guardrails | 8.0 |
| Performance & Resource Management | 6.0 |
| Maintainability & Extensibility | 7.0 |
| Documentation & Comments | 7.5 |
| **Testing Coverage** | **0.5** ← critical gap |
| Dependency Management | 7.0 |
| Compliance & Audit Trail | 8.5 |

---

## 1. Code Structure & Modularity — 8/10

**Strengths**
- Clean pipeline separation: each stage has its own module (`loader`, `chunker`, `embedder`, `store`, `generator`)
- Guardrails module isolates all validation into dedicated classes — `FileGuardrails`, `InputGuardrails`, `RetrievalGuardrails`, `OutputGuardrails`
- Dual-interface design (`app.py` web UI, `main.py` CLI) is a solid separation

**Gaps**
- `app.py` (351 lines) mixes UI rendering, guardrail orchestration, and cache management — violates single-responsibility
- No abstraction between `generator.py` and `generator_local.py` — swapping requires changing imports manually
- Constants scattered across files (`COLLECTION_NAME` in `store.py:17`, `MODEL_NAME` in `embedder.py:15`, `generator_local.py:14`) — no central config

**Fix:** Create `config.py` centralising all constants, and a `rag/pipeline.py` class that owns the 6 guardrail stages so `app.py` becomes a thin view layer.

---

## 2. Readability & Naming — 8.5/10

**Strengths**
- Descriptive function names: `load_file`, `embed_one`, `chunk_text`, `filter_relevant`
- Section dividers (`── ingest ──────────────────`) aid visual scanning throughout `app.py`
- Type hints consistent across all public functions (`guardrails.py:21-28`, `store.py:32-37`, `generator.py:13-21`)

**Gaps**
- Short abbreviations unexplained: `uf` (`app.py:76`) for uploaded file, `rfind` usage in `chunker.py:29`
- `_MAGIC` tuple format in `guardrails.py:85-88` (magic bytes, allowed exts) not documented
- Colour badge logic in `app.py:333-344` uses repeated string literals — no constants

**Fix:** Replace abbreviations with full names, add a comment above `_MAGIC` explaining the (bytes, set) tuple structure.

---

## 3. Error Handling & Resilience — 6.5/10

**Strengths**
- `FileNotFoundError` raised explicitly in `loader.py:17` with a clear message
- `finally` block in `app.py:149` always deletes the tempfile, even on failure
- `loader.py:46` uses `errors="replace"` to survive bad encodings

**Gaps**
- No timeout on `embedder.embed()` (`embedder.py:21`) or `_get_pipeline()` (`generator_local.py:20`) — both can hang indefinitely
- ChromaDB calls (`store.py:50-74`, `store.py:32-46`) have no exception handling — `chromadb.errors.*` propagates raw
- Audit log write in `audit_logger.py:32-35` has no try-except — disk full silently kills the process
- `generator.py:28` creates a new Anthropic client on every call instead of reusing one

**Fix:** Add a 30-second timeout wrapper around embeddings/generation; catch `chromadb.errors.*` in `store.py`; wrap `audit_logger._write` in try-except with a stderr fallback.

---

## 4. Security & Guardrails — 8/10

**Strengths**
- 8-pattern PII scanner covers email, phone, SSN, credit card, IP, passport, Aadhaar, PAN (`guardrails.py:46-55`)
- Magic-byte MIME verification prevents extension spoofing (`guardrails.py:125-146`)
- Path-traversal check on filenames (`guardrails.py:117-122`)
- 14 prompt-injection patterns cover jailbreak, DAN, XML `<system>`, Llama `[INST]` markers (`guardrails.py:167-182`)
- Audit logging of all violations (`audit_logger.py:94-104`)

**Gaps**
- PII regex for IP address (`\d{1,3}` × 4) generates false positives on version strings like `1.0.0.0`
- File bytes written to disk (`app.py:83-85`) before file validation runs — validate first, then write
- No rate limiting — one user can repeatedly ingest large files to exhaust disk/CPU
- Guardrail *warnings* (e.g., PII in query) are displayed but not enforced — user proceeds regardless

**Fix:** For production replace regex PIIDetector with `presidio-analyzer`; move `file_guardrails.validate()` before `tmp.write(file_bytes)`.

---

## 5. Performance & Resource Management — 6/10

**Strengths**
- `@st.cache_resource` on both models prevents reloading per rerun (`app.py:37-43`, `generator_local.py:19-21`)
- Batch embedding for all chunks in a single call (`store.add` → `embedder.embed`)
- ChromaDB HNSW index (`store.py:27`) gives sublinear similarity search
- Context truncation in `generator_local.py:28-36` prevents token overflow

**Gaps**
- `st.session_state["messages"]` grows indefinitely — long sessions accumulate MB of chunk text with no cap
- `audit.log` has no rotation — grows forever (flagged in `MEMORY_MANAGEMENT.md:208`)
- `store.list_sources()` (`store.py:76-81`) loads all metadata into memory — slow at 10k+ chunks
- `embedder.py:24` converts numpy array to Python list unnecessarily on every call — redundant serialisation
- No caching of repeated identical queries

**Fix:** Cap `st.session_state.messages` at 50 entries; switch `audit_logger` to `RotatingFileHandler` (5 MB × 5 files); return numpy array directly from `embedder.embed()` and convert only at the ChromaDB boundary.

---

## 6. Maintainability & Extensibility — 7/10

**Strengths**
- Adding a new file type requires only one new `_load_*` function in `loader.py` — pattern is clear
- PII and injection patterns are data-driven dicts/lists, not hardcoded conditionals
- `GuardrailResult.merge()` makes composing results clean (`guardrails.py:30-35`)

**Gaps**
- Adding a new guardrail layer requires editing `app.py` directly — no plugin/hook system
- Swapping embedding models silently breaks existing ChromaDB indices (vectors from different models are incompatible with no version check)
- `chunk_size` and `overlap` default values appear in both `chunker.py:11` and `main.py:71` — two sources of truth
- No feature flags to adjust guardrail strictness (e.g., disable PII warnings for internal-only deployments)

**Fix:** Add an `embedding_model_version` field to ChromaDB collection metadata; raise an error if the loaded collection was built with a different model.

---

## 7. Documentation & Comments — 7.5/10

**Strengths**
- Module-level docstrings on every file explaining purpose and flow
- `README.md` covers setup, usage, architecture diagram, and concept table
- `MEMORY_MANAGEMENT.md` with full zone breakdown and lifecycle diagram
- Inline comments at non-obvious points (`chunker.py:27-31`, `guardrails.py:117-122`)

**Gaps**
- No docstrings on `__init__` methods — `Embedder` (`embedder.py:17`), `VectorStore` (`store.py:22`), all guardrail classes
- `MAX_INPUT_CHARS = 1800` (`generator_local.py:15`) has no comment explaining how 1800 chars maps to 512 tokens
- `RELEVANCE_THRESHOLD = 0.75` (`guardrails.py:230`) has no tuning guide or empirical justification
- `app.py` has no module docstring explaining its 6 guardrail integration points

**Fix:** Add a one-line comment on every magic number with its derivation; add `__init__` docstrings.

---

## 8. Testing Coverage — 0.5/10

**Strengths**
- Code is structured in a way that makes unit testing straightforward
- Guardrail classes have no external dependencies — can be tested with plain pytest

**Gaps**
- **No test files exist anywhere in the repo**
- `chunker.py` sliding-window logic untested for edge cases (text shorter than chunk, overlap ≥ chunk size)
- `loader.py` untested for corrupted PDFs, empty sheets, mixed-encoding CSVs
- `guardrails.py` injection patterns untested — no verification that patterns match intended strings and don't over-block
- `store.py` ChromaDB operations untested — no tests for add/query/clear cycle
- `audit_logger.py` untested for concurrent writes, disk-full, malformed JSON
- No integration tests for the full ingest → query → generate pipeline

**Fix:** Create `tests/` with the following files to start:
```
tests/
├── test_chunker.py       # boundary conditions
├── test_loader.py        # file format edge cases
├── test_guardrails.py    # injection patterns, PII detection, threshold
└── test_store.py         # ChromaDB integration with tmp directory
```

---

## 9. Dependency Management — 7/10

**Strengths**
- All direct dependencies declared in `requirements.txt`
- `numpy<2` pinned with clear comment explaining the torch compatibility reason
- Lightweight library choices (no heavy frameworks required directly)

**Gaps**
- Lower-bound-only pins (`>=`) allow version drift — `streamlit>=1.35.0` could install a future breaking major version
- No dev dependencies declared (pytest, black, mypy, flake8 missing)
- No lock file (`requirements.txt` is loose — two installs on different days may differ)
- Minimum Python version not documented (code uses walrus operator `:=` → requires 3.8+)

**Fix:**
```
# requirements.txt: add upper bounds
anthropic>=0.40.0,<1.0
chromadb>=0.5.0,<1.0
streamlit>=1.35.0,<2.0
numpy>=1.24,<2.0
```
Create `requirements-dev.txt` with pytest, black, mypy.

---

## 10. Compliance & Audit Trail — 8.5/10

**Strengths**
- JSONL format with UTC ISO 8601 timestamps (`audit_logger.py:33`)
- Query text and answer content never written to log — privacy by design (`audit_logger.py:52`, `:68`)
- PII scrubbing applied to all log values before write (`audit_logger.py:20-29`)
- Five event types cover the full pipeline: `FILE_INGESTED`, `QUERY`, `ANSWER_GENERATED`, `GUARDRAIL_VIOLATION`, `GUARDRAIL_WARNING`
- Visual audit log tab in app with colour-coded events and summary metrics (`app.py:305-350`)

**Gaps**
- No log rotation or retention policy — disk usage unbounded
- No log integrity mechanism (HMAC signature) — logs can be edited without detection
- No user/session identifier in log entries — cannot attribute actions to a specific user
- Guardrail PII scrub regex (`audit_logger.py:21-25`) is weaker than the full PIIDetector patterns in `guardrails.py`

**Fix:** Add HMAC-SHA256 signature field to each log entry; implement `RotatingFileHandler`; include a `session_id` (UUID generated at session start) in every event.

---

## Top 5 Issues — Ranked by Impact

### 1. Zero test coverage (Testing: 0.5/10)
No tests exist. Guardrail patterns, chunker boundary conditions, loader edge cases, and ChromaDB operations are all unverified.
**Start with:** `tests/test_guardrails.py` — pure Python, no mocking needed.

### 2. Unbounded resource consumption (Performance: 6/10)
`st.session_state["messages"]` grows without limit; `audit.log` never rotates.
**Fix in:** `app.py` (add message cap) and `audit_logger.py` (add `RotatingFileHandler`).

### 3. No error handling on critical I/O paths (Error Handling: 6.5/10)
ChromaDB, model inference, and audit log writes can all fail silently or crash the app.
**Fix in:** `store.py`, `embedder.py`, `generator_local.py`, `audit_logger.py`.

### 4. Monolithic app.py (Structure: 8/10)
UI, guardrail orchestration, and Streamlit cache management are all coupled in one 351-line file.
**Fix:** Extract `rag/pipeline.py` — a `RAGPipeline` class that owns the full ingest and query flow.

### 5. Loose dependency pins (Dependencies: 7/10)
`>=` pins allow breaking version upgrades on fresh installs; no lock file; no dev deps.
**Fix:** Add upper bounds, create `requirements-dev.txt`, document Python ≥ 3.8 requirement.
