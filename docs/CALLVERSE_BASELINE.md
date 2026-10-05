# CallVerse Baseline

## Base Project

CallVerse starts from the [HelpPilot](https://github.com/poysa213/HelpPilot)
repository. HelpPilot is an MIT-licensed customer-support agent that triages customer
messages, uses order and policy tools, drafts grounded replies, and pauses proposed
refunds for staff approval.

The upstream MIT license and copyright notice are preserved verbatim in `LICENSE`.
The inherited Python package remains named `helppilot` in this baseline so that working
imports are not disrupted.

## What We Inherit

- A five-node LangGraph workflow for triage, solving, approval, review, and response
- Chroma vector retrieval with sentence-transformer embeddings, a cross-encoder
  reranker, and source citations
- Customer, order, tracking, refund-policy, refund, escalation, and approval tools
- SQLite storage for sample business data, tickets, approvals, logs, and durable
  LangGraph checkpoints
- A Streamlit customer-chat and staff-approval interface
- Human approval before refund execution and escalation paths for unsupported cases
- A 30-ticket evaluation utility with correctness, groundedness, escalation, safety,
  latency, and estimated-cost measurements

## What CallVerse Will Add Later

- Client Simulator
- Support-center simulation
- KPI engine
- Quality Analyst extensions
- Demand forecasting
- Workforce Manager
- Reinforcement-learning comparison
- Scenario studio

These capabilities are roadmap items only and are not implemented in this baseline.

## Baseline Verification

- **Python version:** CPython 3.11.4 in `.venv`, provisioned and managed by `uv`
- **Installation status:** `uv sync --python 3.11 --locked` installed all 134 locked
  packages successfully
- **Database initialization:** `python -m helppilot.seed` completed with 5 customers,
  5 tickets, 5 orders (1 lost), and 6 stored facts
- **RAG/index initialization:** the seed built 11 Chroma chunks in `helppilot_kb`; the
  documented retrieval command loaded the embedding and reranker models and returned
  relevant policy citations
- **Application startup:** Streamlit started on port 8502 and its health endpoint
  returned HTTP 200 with `ok`; Streamlit's local app test also executed `app.py` with
  zero uncaught exceptions
- **Test status:** no local unit-test files or test framework configuration were found;
  import and bytecode-compilation smoke checks passed. The 30-ticket `eval.py`
  evaluation was intentionally skipped because it requires `GROQ_API_KEY` and makes
  paid external LLM calls
- **Configuration:** `GROQ_API_KEY` is required for chat and evaluation. LangSmith
  tracing is optional and uses `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and
  `LANGSMITH_ENDPOINT`. Place real values only in the ignored `.env`; `.env.example`
  contains placeholders
- **Known problems:** without `GROQ_API_KEY`, the UI starts but intentionally displays
  a configuration error and cannot process chat messages. Hugging Face downloads work
  without a token but warn about lower rate limits and degraded, non-symlink caching on
  Windows. The upstream `LICENSE` copyright line is literally
  `Copyright (c) 2026 <Your Name>` and has been preserved verbatim rather than silently
  altering upstream attribution
