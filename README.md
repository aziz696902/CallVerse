# HelpPilot

A small customer-support agent built with LangGraph. It reads a customer message,
looks up their order, searches the help docs, and writes a grounded reply. When it
wants to do something sensitive — like give a refund — it stops and waits for a human
to approve.

I built this to learn how to design an agent that is safe and easy to follow, not
just one that answers questions. Every run shows up as a single trace in LangSmith.

```
customer → triage → solver → approval → reviewer → respond → reply
           (20B)    (120B)   (human)    (120B)     (code)
```

## What it does

Ask it *"my package never arrived, order ORD-5001"* and it will:

1. decide the message needs tools (triage),
2. look up the order and tracking, find it is lost, read the refund policy (solver),
3. draft a refund and **pause for a human to approve it** (approval),
4. check the reply is backed by the docs and has citations (reviewer),
5. send the reply and save everything (respond).

## The 5 steps

| Step | Uses an LLM? | What it is for |
|------|------|----------------|
| **triage** | yes (small model) | Sort the message: simple answer, use tools, or send to a human. A cheap first pass. |
| **solver** | yes (big model) | The main worker. Searches docs, calls tools, and *proposes* a reply and any action. It only drafts refunds — it never sends money. |
| **approval** | no | Stops the run when a refund is proposed and waits for a person to approve or reject. |
| **reviewer** | yes (big model) | Last check before sending: is the reply supported by the docs and cited? If not, it sends the work back to the solver (up to 2 times). |
| **respond** | no | Cleans up: remove personal data, add citations, save to the database, send. |

3 of the 5 steps use an LLM. Only 2 of them — solver and reviewer — use the big
model to do real reasoning; triage uses a small model for a quick sort. The other
two steps (approval and respond) are plain, predictable code. That keeps the agent
easy to trust and easy to read.

## Two people, not one

A customer should never approve their own refund, and they should not have to wait.
So the agent works like a real support team:

- The **customer** chats. When the agent proposes a refund, the customer just sees
  *"a specialist is reviewing this, we'll get back to you."* The chat ends there.
- A **staff member** opens a separate view with a queue of pending refunds. They see
  what the agent proposed and click Approve or Reject.
- When staff decide, the answer is sent back to the customer.

This is possible because of LangGraph's `interrupt()`. It saves the whole run to a
SQLite file and stops. Later, a call to `resume_turn(thread_id, approved=...)` picks
it up from the exact same place — even from a different screen or process. Nothing is
lost and nobody is blocked.

In the demo, both roles are in one app behind a sidebar switch, so you can try both
sides yourself.

## RAG (searching the docs)

The document search is real (only the order/tracking data is fake):

1. About 8 policy documents are split into chunks and stored in Chroma.
2. A query pulls the 8 closest chunks.
3. A cross-encoder reranks them and keeps the best 3. This model reads the question
   and the passage together, so it is more accurate than plain similarity.
4. Each chunk keeps its id, which becomes a citation like `[refund-policy]`.

You can try the search on its own:

```bash
python -m helppilot.rag "my package never arrived, can I get a refund?"
```

## Tech

- **LangGraph** for the agent, with a SQLite checkpointer for the pause/resume.
- **Groq** for the models (`gpt-oss-20b` for triage, `gpt-oss-120b` for solving and review).
- **Chroma** + a **cross-encoder reranker** for search.
- **SQLite** for the data (customers, orders, tickets, logs, approvals).
- **Streamlit** for the UI.
- **LangSmith** for tracing.

## Run it

You need Python 3.11+ and a [Groq API key](https://console.groq.com/keys).
A [LangSmith](https://smith.langchain.com/) key is optional (it adds tracing).

```bash
# install
uv sync                          # or: pip install -r requirements.txt

# add your keys
cp .env.example .env             # then paste your GROQ_API_KEY

# load the sample data and build the search index
uv run python -m helppilot.seed

# start the app
uv run streamlit run app.py
```

Then, in the app:

1. As **Customer** (Alice), type *"my package never arrived, order ORD-5001"*.
2. Switch to **Staff** in the sidebar and approve the refund.
3. Switch back to **Customer** to see the result.

## Evaluation

`python eval.py` runs about 30 test tickets through the agent and checks:

- is the answer correct (judged by an LLM),
- is it grounded in the retrieved docs,
- does it escalate the right cases,
- and one strict rule: **no refund is ever sent without approval.**

It also measures latency and estimated cost, prints a table, and writes the result
to `EVAL_RESULTS.md`.

On the last run (30 tickets): **86.7% correct, 100% grounded, 100% correct
escalations, and 100% of refunds went through approval** — at about $0.0008 per
ticket. See [`EVAL_RESULTS.md`](EVAL_RESULTS.md) for the full table.

## Project structure

```
app.py                 Streamlit UI (customer chat + staff approval queue)
eval.py                evaluation script
helppilot/
  config.py            model names, paths, tracing setup
  db.py                SQLite schema and helpers
  seed.py              load sample data + build the search index
  kb_docs.py           the policy/FAQ documents
  rag.py               search + rerank + citations
  tools.py             the agent's tools (order lookup, refund, etc.)
  graph.py             the 5-step LangGraph
  eval_dataset.py      the test tickets
```

## What I left out (on purpose)

To keep the project small and clear, I did not build: email/Slack channels, hybrid
search, background workers, billing, teams, or login. These would add size without
making the core agent easier to understand.
