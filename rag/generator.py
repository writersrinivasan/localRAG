"""
Generator — sends the retrieved context + question to Claude and returns an answer.

RAG flow:
  1. Embed the question
  2. Retrieve the top-k most similar chunks from the vector store  ← done in main.py
  3. Build a prompt: "Here is context: ... Answer: <question>"     ← done here
  4. Call Claude and return the response
"""

import os
import anthropic
from typing import List, Dict, Any


def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer using Claude with the retrieved chunks as context.

    context_chunks: list of {text, source, page} dicts from the vector store
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
