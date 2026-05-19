"""
Document loader — converts files into plain text.

Supported: .txt, .md, .pdf, .docx, .xlsx, .xls, .csv, .pptx
Each loader returns a list of page/sheet dicts: {text, source, page}
"""

import os
from pathlib import Path
from typing import List, Dict, Any


def load_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a file and return a list of {text, source, page} dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loaders = {
        ".txt": _load_text,
        ".md":  _load_text,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".doc":  _load_docx,
        ".xlsx": _load_spreadsheet,
        ".xls":  _load_spreadsheet,
        ".csv":  _load_csv,
        ".pptx": _load_pptx,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders)}")

    pages = loader(str(path))
    # Attach source filename to every page
    for p in pages:
        p["source"] = path.name
    return pages


# ── individual loaders ────────────────────────────────────────────────────────

def _load_text(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [{"text": f.read(), "page": 1}]


def _load_pdf(path: str) -> List[Dict[str, Any]]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i})
    return pages


def _load_docx(path: str) -> List[Dict[str, Any]]:
    from docx import Document
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "page": 1}]


def _load_spreadsheet(path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    xl = pd.ExcelFile(path)
    pages = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        # Convert each sheet to a readable text table
        text = f"Sheet: {sheet}\n{df.to_string(index=False)}"
        pages.append({"text": text, "page": sheet})
    return pages


def _load_csv(path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path)
    return [{"text": df.to_string(index=False), "page": 1}]


def _load_pptx(path: str) -> List[Dict[str, Any]]:
    from pptx import Presentation
    prs = Presentation(path)
    pages = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            pages.append({"text": "\n".join(texts), "page": i})
    return pages
