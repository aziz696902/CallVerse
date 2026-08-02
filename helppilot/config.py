"""Central configuration: model IDs, filesystem paths, and tracing setup.

Everything that another module might need to know about "where things live" or
"which model to call" is resolved here exactly once, so the rest of the codebase
never touches os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# NOTE: the HF token-path workaround for stale ~/.cache/huggingface tokens lives in
# helppilot/__init__.py, so it runs before huggingface_hub is imported anywhere.

# Load .env from the project root (parent of this package) as early as possible.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Model IDs (pinned per spec; do not substitute) ---------------------------
# A small fast model for triage, a larger one for solving and review.
TRIAGE_MODEL = "openai/gpt-oss-20b"
SOLVER_MODEL = "openai/gpt-oss-120b"
REVIEWER_MODEL = "openai/gpt-oss-120b"

# --- Retrieval / reranking models ---------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers bi-encoder for Chroma
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder reranker

# --- Secrets ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Filesystem paths ---------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "helppilot.db"           # business data (customers, orders, ...)
CHROMA_DIR = DATA_DIR / "chroma"              # persistent vector store
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"  # LangGraph durable state

# Chroma collection name for the policy/FAQ knowledge base.
CHROMA_COLLECTION = "helppilot_kb"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def setup_tracing() -> bool:
    """Ensure LangSmith env vars are set so every LangChain/LangGraph run traces.

    Returns True if tracing is enabled and an API key is present. Reading .env via
    load_dotenv() above already populates os.environ; this just normalizes the
    flag and reports status so callers can print a helpful message.
    """
    tracing_on = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    if tracing_on and has_key:
        # LangChain reads these names; make sure they're present for child procs.
        os.environ.setdefault("LANGSMITH_PROJECT", "helppilot")
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        return True
    return False
