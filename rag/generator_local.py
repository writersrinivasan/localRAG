"""
Local generator — uses google/flan-t5-base for answer generation.
No API key needed. Model (~250 MB) downloads once on first use.

flan-t5 is an instruction-tuned T5 model good at reading-comprehension style Q&A:
  given context + question → produces a grounded answer
"""

from typing import List, Dict, Any
from transformers import pipeline
import streamlit as st


MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_CHARS = 1800   # stay within flan-t5's 512-token input limit
MAX_NEW_TOKENS  = 200


@st.cache_resource(show_spinner="Loading local LLM (flan-t5-base)…")
def _get_pipeline():
    return pipeline("text2text-generation", model=MODEL_NAME)


def generate_answer_local(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer using flan-t5-base with retrieved chunks as context."""
    # Combine chunks; truncate so we don't overflow the model's token limit
    context_parts = []
    total = 0
    for chunk in context_chunks:
        text = chunk["text"]
        if total + len(text) > MAX_INPUT_CHARS:
            text = text[: MAX_INPUT_CHARS - total]
            context_parts.append(text)
            break
        context_parts.append(text)
        total += len(text)

    context = " ".join(context_parts)

    prompt = (
        f"Answer the question based only on the context below.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    pipe = _get_pipeline()
    result = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return result[0]["generated_text"].strip()
