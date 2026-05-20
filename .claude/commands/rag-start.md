# RAG Environment Start & Health Check

Run the full localRAG environment check and start the Streamlit UI.

## Steps to execute

1. **Check Python and key packages**
   ```bash
   python3 -c "import streamlit, chromadb, sentence_transformers, transformers, pypdf, docx, pandas, numpy; print('All packages OK')"
   ```
   If any import fails, tell the user which package is missing and suggest: `pip3 install -r requirements.txt`

2. **Run the security test suite**
   ```bash
   cd /Volumes/SRINI/AppCreation/simpleRAG && python3 -m pytest tests/ -q --tb=short 2>&1 | tail -20
   ```
   Report: total tests, passed, failed. If any fail, show the failing test names.

3. **Check for a running Streamlit process**
   ```bash
   pgrep -fl "streamlit run app" 2>/dev/null
   ```
   - If already running, report the port (check 8501 then 8502) and skip step 4.
   - If not running, proceed to step 4.

4. **Start Streamlit** (only if not already running)
   ```bash
   pkill -f "streamlit run app" 2>/dev/null; sleep 1
   nohup /Users/srinivasanramanujam/Library/Python/3.9/bin/streamlit run /Volumes/SRINI/AppCreation/simpleRAG/app.py --server.headless true > /tmp/streamlit_rag.log 2>&1 &
   echo "Started PID $!"
   ```
   Then wait 5 seconds and verify it responded:
   ```bash
   sleep 5 && curl -s http://localhost:8501 -o /dev/null -w "%{http_code}"
   ```

5. **Check log files exist and are writable**
   ```bash
   ls -lh /Volumes/SRINI/AppCreation/simpleRAG/audit.log /Volumes/SRINI/AppCreation/simpleRAG/rag_diagnostics.log 2>/dev/null
   ```

6. **Check ChromaDB data directory**
   ```bash
   ls /Volumes/SRINI/AppCreation/simpleRAG/chroma_db/ 2>/dev/null | head -5 || echo "Empty — no documents ingested yet"
   ```

## Report format

Summarise with a status table:

| Check | Status |
|---|---|
| Packages | ✅ / ❌ missing: X |
| Tests | ✅ N passed / ❌ N failed |
| Streamlit | ✅ running on port XXXX / 🚀 started |
| Log files | ✅ present / ⚠️ missing |
| ChromaDB | ✅ N collections / 📭 empty |

End with: **App is ready at http://localhost:8501** (or 8502 if that port is in use).
