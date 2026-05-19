"""
Simple RAG System — CLI entry point

Commands:
  python main.py ingest <file_or_folder>   Add documents to the knowledge base
  python main.py query  "<question>"       Ask a question
  python main.py list                      Show ingested documents
  python main.py clear                     Wipe the knowledge base
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import Embedder
from rag.store import VectorStore
from rag.generator import generate_answer

load_dotenv()
console = Console()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc",
                        ".xlsx", ".xls", ".csv", ".pptx"}


# ── ingest ────────────────────────────────────────────────────────────────────

def cmd_ingest(path: str):
    p = Path(path)
    files = []

    if p.is_dir():
        for f in p.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
    elif p.is_file():
        files = [p]
    else:
        console.print(f"[red]Path not found:[/red] {path}")
        sys.exit(1)

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    console.print(f"\n[bold]Loading sentence-transformer model...[/bold]")
    embedder = Embedder()
    store = VectorStore()

    total_chunks = 0
    for file in files:
        console.print(f"\n[cyan]Processing:[/cyan] {file.name}")
        try:
            pages = load_file(str(file))
        except Exception as e:
            console.print(f"  [red]Failed to load:[/red] {e}")
            continue

        file_chunks = []
        file_meta = []
        for page in pages:
            chunks = chunk_text(page["text"], chunk_size=500, overlap=100)
            for i, chunk in enumerate(chunks):
                file_chunks.append(chunk)
                file_meta.append({
                    "source": file.name,
                    "page": str(page["page"]),
                    "chunk_index": i,
                })

        if not file_chunks:
            console.print("  [yellow]No text extracted.[/yellow]")
            continue

        console.print(f"  Generating embeddings for {len(file_chunks)} chunks...")
        embeddings = embedder.embed(file_chunks)
        added = store.add(file_chunks, embeddings, file_meta)
        total_chunks += added
        console.print(f"  [green]✓[/green] Stored {added} chunks")

    console.print(
        Panel(f"[green]Done![/green] Added [bold]{total_chunks}[/bold] chunks. "
              f"Total in store: [bold]{store.count()}[/bold]",
              title="Ingest Complete")
    )


# ── query ─────────────────────────────────────────────────────────────────────

def cmd_query(question: str, top_k: int = 5):
    store = VectorStore()
    if store.count() == 0:
        console.print("[yellow]Knowledge base is empty. Run: python main.py ingest <file>[/yellow]")
        return

    console.print("\n[bold]Searching knowledge base...[/bold]")
    embedder = Embedder()
    query_embedding = embedder.embed_one(question)
    hits = store.query(query_embedding, n_results=top_k)

    # Show retrieved context (the "R" in RAG)
    console.print(f"\n[dim]Retrieved {len(hits)} relevant chunks:[/dim]")
    for i, h in enumerate(hits, 1):
        console.print(
            f"  [dim]{i}. {h['source']} p.{h['page']} (similarity distance: {h['distance']})[/dim]"
        )

    console.print("\n[bold]Generating answer...[/bold]")
    answer = generate_answer(question, hits)

    console.print(Panel(answer, title=f"[bold green]Answer[/bold green]", border_style="green"))


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list():
    store = VectorStore()
    sources = store.list_sources()

    if not sources:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    table = Table(title=f"Knowledge Base ({store.count()} total chunks)")
    table.add_column("Document", style="cyan")
    for s in sources:
        table.add_row(s)
    console.print(table)


# ── clear ─────────────────────────────────────────────────────────────────────

def cmd_clear():
    confirm = input("This will delete all ingested documents. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        store = VectorStore()
        store.clear()
        console.print("[green]Knowledge base cleared.[/green]")
    else:
        console.print("Aborted.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simple RAG system — ingest docs and query them with Claude"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Add documents to the knowledge base")
    p_ingest.add_argument("path", help="File or directory to ingest")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Your question in quotes")
    p_query.add_argument("--top-k", type=int, default=5,
                         help="Number of chunks to retrieve (default: 5)")

    sub.add_parser("list", help="Show ingested documents")
    sub.add_parser("clear", help="Wipe the knowledge base")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args.path)
    elif args.command == "query":
        cmd_query(args.question, args.top_k)
    elif args.command == "list":
        cmd_list()
    elif args.command == "clear":
        cmd_clear()


if __name__ == "__main__":
    main()
