"""
Local generator — uses google/flan-t5-base for answer generation.
No API key needed. Model (~250 MB) downloads once on first use.

flan-t5 is an instruction-tuned T5 model good at reading-comprehension style Q&A:
  given context + question → produces a grounded answer
"""

import concurrent.futures
from typing import List, Dict, Any

from .exceptions import GeneratorError, GeneratorTimeoutError

MODEL_NAME      = "google/flan-t5-base"
MAX_INPUT_CHARS = 1800   # ~512 tokens for flan-t5's encoder limit
MAX_NEW_TOKENS  = 200
INFERENCE_TIMEOUT_SECONDS = 120


def _load_pipeline():
    """Load and return the flan-t5 pipeline. Called once and cached by Streamlit."""
    try:
        from transformers import pipeline
        return pipeline("text2text-generation", model=MODEL_NAME)
    except ImportError as exc:
        raise GeneratorError(
            "transformers is not installed. Run: pip install transformers"
        ) from exc
    except Exception as exc:
        raise GeneratorError(
            f"Failed to load generation model '{MODEL_NAME}': {exc}. "
            "Check your internet connection or HuggingFace cache."
        ) from exc


# Lazy singleton — avoids importing streamlit when used outside the UI
_pipeline_instance = None

def _get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = _load_pipeline()
    return _pipeline_instance


def generate_answer_local(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer using flan-t5-base with retrieved chunks as context."""
    if not question or not question.strip():
        raise GeneratorError("Question must not be empty.")
    if not context_chunks:
        raise GeneratorError("No context chunks provided — cannot generate an answer.")

    # Build context, truncate to stay within token limit
    context_parts = []
    total = 0
    for chunk in context_chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        remaining = MAX_INPUT_CHARS - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        context_parts.append(text)
        total += len(text)

    if not context_parts:
        raise GeneratorError("All context chunks were empty — cannot generate an answer.")

    context = " ".join(context_parts)
    prompt = (
        f"Answer the question based only on the context below.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    pipe = _get_pipeline()

    def _run():
        return pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            result = future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        raise GeneratorTimeoutError(
            f"Generation timed out after {INFERENCE_TIMEOUT_SECONDS} s. "
            "The model may be busy — please try again."
        )
    except GeneratorError:
        raise
    except Exception as exc:
        raise GeneratorError(f"Generation failed: {exc}") from exc

    if not result or not isinstance(result, list) or not result[0].get("generated_text"):
        raise GeneratorError("Model returned an empty or malformed response.")

    answer = result[0]["generated_text"].strip()
    if not answer:
        raise GeneratorError("Model returned a blank answer.")

    return answer
