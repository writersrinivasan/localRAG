"""
Simple RAG System — CLI entry point

Commands:
  python main.py ingest <file_or_folder>   Add documents to the knowledge base
  python main.py query  "<question>"       Ask a question
  python main.py list                      Show ingested documents
  python main.py clear                     Wipe the knowledge base
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import VectorStore
from rag.generator import generate_answer
from rag.exceptions import (
    RAGError, LoaderError, ChunkerError,
    EmbedderError, EmbedderTimeoutError,
    StoreError, StoreUnavailableError,
    GeneratorError, GeneratorTimeoutError, GeneratorAPIError,
)

load_dotenv()
console = Console()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc",
                        ".xlsx", ".xls", ".csv", ".pptx"}


# ── shared initialisation ─────────────────────────────────────────────────────

def _init_store() -> VectorStore:
    try:
        return VectorStore()
    except StoreUnavailableError as exc:
        console.print(f"[red]Vector store unavailable:[/red] {exc}")
        sys.exit(1)


def _init_embedder() -> Embedder:
    try:
        console.print("[bold]Loading embedding model…[/bold]")
        return Embedder()
    except EmbedderError as exc:
        console.print(f"[red]Embedding model failed to load:[/red] {exc}")
        sys.exit(1)


# ── ingest ────────────────────────────────────────────────────────────────────

def cmd_ingest(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    elif p.is_file():
        files = [p]
    else:
        console.print(f"[red]Path not found:[/red] {path}")
        sys.exit(1)

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    embedder = _init_embedder()
    store    = _init_store()
    total    = 0

    for file in files:
        console.print(f"\n[cyan]Processing:[/cyan] {file.name}")

        # parse
        try:
            pages = load_file(str(file))
        except LoaderError as exc:
            console.print(f"  [red]Cannot read file:[/red] {exc}")
            continue

        # chunk
        chunks, meta = [], []
        try:
            for page in pages:
                for i, c in enumerate(chunk_text(page["text"])):
                    chunks.append(c)
                    meta.append({
                        "source":      file.name,
                        "page":        str(page["page"]),
                        "chunk_index": i,
                    })
        except ChunkerError as exc:
            console.print(f"  [red]Chunking failed:[/red] {exc}")
            continue

        if not chunks:
            console.print("  [yellow]No text extracted — skipping.[/yellow]")
            continue

        # embed
        console.print(f"  Embedding {len(chunks)} chunks…")
        try:
            embeddings = embedder.embed(chunks)
        except EmbedderTimeoutError as exc:
            console.print(f"  [red]Embedding timed out:[/red] {exc}")
            continue
        except EmbedderError as exc:
            console.print(f"  [red]Embedding failed:[/red] {exc}")
            continue

        # store
        try:
            added  = store.add(chunks, embeddings, meta)
            total += added
            console.print(f"  [green]✓[/green] Stored {added} chunks")
        except StoreError as exc:
            console.print(f"  [red]Storage failed:[/red] {exc}")
            continue

    try:
        count = store.count()
    except StoreError:
        count = "?"

    console.print(Panel(
        f"[green]Done![/green] Added [bold]{total}[/bold] chunks. "
        f"Total in store: [bold]{count}[/bold]",
        title="Ingest Complete",
    ))


# ── query ─────────────────────────────────────────────────────────────────────

def cmd_query(question: str, top_k: int = 5) -> None:
    store = _init_store()

    try:
        count = store.count()
    except StoreError as exc:
        console.print(f"[red]Vector store error:[/red] {exc}")
        sys.exit(1)

    if count == 0:
        console.print("[yellow]Knowledge base is empty. "
                      "Run: python main.py ingest <file>[/yellow]")
        return

    embedder = _init_embedder()

    # embed query
    try:
        console.print("\n[bold]Embedding question…[/bold]")
        query_vec = embedder.embed_one(question)
    except EmbedderTimeoutError as exc:
        console.print(f"[red]Embedding timed out:[/red] {exc}")
        sys.exit(1)
    except EmbedderError as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        sys.exit(1)

    # retrieve
    try:
        hits = store.query(query_vec, n_results=top_k)
    except StoreError as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        sys.exit(1)

    if not hits:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return

    console.print(f"\n[dim]Retrieved {len(hits)} chunks:[/dim]")
    for i, h in enumerate(hits, 1):
        console.print(
            f"  [dim]{i}. {h['source']} p.{h['page']} "
            f"(distance: {h['distance']})[/dim]"
        )

    # generate
    try:
        console.print("\n[bold]Generating answer…[/bold]")
        answer = generate_answer(question, hits)
    except GeneratorTimeoutError as exc:
        console.print(f"[red]Generation timed out:[/red] {exc}")
        sys.exit(1)
    except GeneratorAPIError as exc:
        console.print(f"[red]API error:[/red] {exc}")
        sys.exit(1)
    except GeneratorError as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        sys.exit(1)

    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    store = _init_store()
    try:
        sources = store.list_sources()
        count   = store.count()
    except StoreError as exc:
        console.print(f"[red]Cannot read knowledge base:[/red] {exc}")
        sys.exit(1)

    if not sources:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    table = Table(title=f"Knowledge Base ({count} total chunks)")
    table.add_column("Document", style="cyan")
    for s in sources:
        table.add_row(s)
    console.print(table)


# ── clear ─────────────────────────────────────────────────────────────────────

def cmd_clear() -> None:
    confirm = input("This will delete all ingested documents. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        console.print("Aborted.")
        return

    store = _init_store()
    try:
        store.clear()
        console.print("[green]Knowledge base cleared.[/green]")
    except StoreError as exc:
        console.print(f"[red]Clear failed:[/red] {exc}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="localRAG — ingest docs and query them locally"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Add documents to the knowledge base")
    p_ingest.add_argument("path", help="File or directory to ingest")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Your question in quotes")
    p_query.add_argument("--top-k", type=int, default=5,
                         help="Chunks to retrieve (default: 5)")

    sub.add_parser("list",  help="Show ingested documents")
    sub.add_parser("clear", help="Wipe the knowledge base")

    args = parser.parse_args()

    try:
        if args.command == "ingest":
            cmd_ingest(args.path)
        elif args.command == "query":
            cmd_query(args.question, args.top_k)
        elif args.command == "list":
            cmd_list()
        elif args.command == "clear":
            cmd_clear()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except RAGError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
