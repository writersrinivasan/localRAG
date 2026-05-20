"""
Generator — sends the retrieved context + question to Claude and returns an answer.

RAG flow:
  1. Embed the question
  2. Retrieve the top-k most similar chunks  ← done upstream
  3. Build a prompt with context             ← here
  4. Call Claude with retry on rate-limit    ← here
"""

import os
import time
from typing import List, Dict, Any

from .exceptions import GeneratorError, GeneratorAPIError

# Shared client — created once, reused across all calls
_client = None

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2   # seconds; doubles on each attempt


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise GeneratorError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )

    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
        return _client
    except ImportError as exc:
        raise GeneratorError(
            "anthropic package is not installed. Run: pip install anthropic"
        ) from exc
    except Exception as exc:
        raise GeneratorError(f"Failed to initialise Anthropic client: {exc}") from exc


def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer using Claude with the retrieved chunks as context."""
    if not question or not question.strip():
        raise GeneratorError("Question must not be empty.")
    if not context_chunks:
        raise GeneratorError("No context chunks — cannot generate an answer.")

    client = _get_client()

    context_parts = [
        f"[Source {i}: {c['source']}, page {c['page']}]\n{c['text']}"
        for i, c in enumerate(context_chunks, start=1)
    ]
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
        "If the context does not contain enough information, say so clearly.\n\n"
        f"CONTEXT:\n{chr(10).join(context_parts)}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:"
    )

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            import anthropic
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip() if message.content else ""
            if not text:
                raise GeneratorError("Claude returned an empty response.")
            return text

        except anthropic.RateLimitError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            continue

        except anthropic.APITimeoutError as exc:
            raise GeneratorAPIError(
                "Claude API request timed out. Check your network and try again."
            ) from exc

        except anthropic.APIConnectionError as exc:
            raise GeneratorAPIError(
                "Cannot reach the Claude API. Check your internet connection."
            ) from exc

        except anthropic.AuthenticationError as exc:
            raise GeneratorAPIError(
                "Invalid ANTHROPIC_API_KEY. Please check your credentials."
            ) from exc

        except anthropic.APIStatusError as exc:
            raise GeneratorAPIError(
                f"Claude API returned an error (HTTP {exc.status_code}): {exc.message}"
            ) from exc

        except GeneratorError:
            raise

        except Exception as exc:
            raise GeneratorError(f"Unexpected error during generation: {exc}") from exc

    raise GeneratorAPIError(
        f"Claude API rate limit exceeded after {MAX_RETRIES} retries. "
        "Please wait a moment and try again."
    ) from last_exc
