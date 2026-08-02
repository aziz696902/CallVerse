"""HelpPilot — an AI customer-support agent (LangGraph + Groq + Chroma)."""

# The embedding and reranker models are public, so we download them anonymously.
# If a local Hugging Face token file is present but stale, it can make even public
# downloads fail with a 401. Unless the user provides their own token, we point the
# token lookup at a path that doesn't exist to force clean anonymous access. This has
# to run before huggingface_hub is imported (langchain_groq imports it), so it lives
# at the top of the package __init__.
import os as _os
from pathlib import Path as _Path

if not (_os.getenv("HF_TOKEN") or _os.getenv("HUGGING_FACE_HUB_TOKEN")):
    _os.environ.setdefault(
        "HF_TOKEN_PATH", str(_Path(__file__).resolve().parent.parent / "data" / ".hf_no_token")
    )
    _os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # Anonymous download is intentional, so hide the "set a HF_TOKEN" warning.
    import logging as _logging

    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

__version__ = "0.1.0"
