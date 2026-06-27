"""Configuration and model factories.

All credentials are read from environment variables (loaded from a local ``.env``
file in development). Nothing is hard-coded — copy ``.env.example`` to ``.env``
and fill in your keys.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Claude model used for reasoning and tool calling. Defaults to the most capable
# Opus tier; set ANTHROPIC_MODEL=claude-sonnet-4-6 for a cheaper, faster option.
CHAT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

# Local sentence-transformers model used to embed documents for RAG. Runs on CPU,
# no API key required.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def require_anthropic_key() -> None:
    """Fail fast with a helpful message if the Anthropic key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
            "Anthropic API key (https://console.anthropic.com/)."
        )


def get_chat_model():
    """Return a configured ChatAnthropic model.

    Note: temperature/top_p are intentionally not set — they are rejected by the
    Opus 4.7+ models, and the defaults are fine for analytical work.
    """
    from langchain_anthropic import ChatAnthropic

    require_anthropic_key()
    return ChatAnthropic(model=CHAT_MODEL, max_tokens=4096)


def get_embeddings():
    """Return a local HuggingFace embedding model (no API key needed)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
