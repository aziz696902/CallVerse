"""Agent tools: mocked externals + one real RAG tool.

Everything the solver can *do* lives here. Externals (orders, tracking, CRM) are
mocked from SQLite so the demo is deterministic and offline-friendly; only
check_refund_policy touches a real model (the Chroma+reranker pipeline).

Two facts about the design:
  * Every tool appends to actions_log, giving the reviewer/UI a faithful audit
    trail of what actually happened during a run.
  * The current thread_id/customer_id are carried in a contextvar (set by the
    graph before it invokes the solver) so @tool functions — which only receive
    their declared args — can still attribute their log rows to the right run.
  * issue_refund is SENSITIVE: the solver never calls it. It proposes a refund via
    create_refund_draft; the graph runs issue_refund itself, but only after the
    human approval interrupt resolves. That separation is what keeps refunds safe.
"""
from __future__ import annotations

import contextvars
import json

from langchain_core.tools import tool

from . import db, rag

# Per-run context so tools can log against the active thread/customer.
_run_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("run_ctx", default={})


def set_run_context(thread_id: str | None, customer_id: str | None) -> None:
    _run_ctx.set({"thread_id": thread_id, "customer_id": customer_id})


def _ctx() -> dict:
    return _run_ctx.get()


def _log(action: str, payload: dict, result: dict) -> None:
    c = _ctx()
    db.log_action(
        action,
        payload=payload,
        result=result,
        thread_id=c.get("thread_id"),
        customer_id=c.get("customer_id"),
    )


# --- Mocked external tools (read from SQLite) ---------------------------------

@tool
def get_order(order_id: str) -> str:
    """Look up an order by id. Returns item, amount, status, and customer."""
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    result = dict(row) if row else {"error": f"No order found with id {order_id}"}
    _log("get_order", {"order_id": order_id}, result)
    return json.dumps(result)


@tool
def get_tracking(order_id: str) -> str:
    """Get carrier tracking status for an order. Reveals if a package is lost."""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT id, status, carrier, tracking_number, last_scan FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
    if not row:
        result = {"error": f"No order found with id {order_id}"}
    else:
        result = {
            "order_id": row["id"],
            "carrier": row["carrier"],
            "tracking_number": row["tracking_number"],
            "status": row["status"],
            "last_scan": row["last_scan"],
            "is_lost": row["status"] == "lost",
        }
    _log("get_tracking", {"order_id": order_id}, result)
    return json.dumps(result)


@tool
def check_refund_policy(query: str) -> str:
    """Search the policy/FAQ knowledge base. Returns passages with [citation] ids.

    This is the REAL retrieval tool: Chroma recall + cross-encoder rerank.
    """
    results = rag.retrieve(query, k=3)
    _log("check_refund_policy", {"query": query}, {"doc_ids": [r["doc_id"] for r in results]})
    return json.dumps({"results": results})


@tool
def create_refund_draft(order_id: str, amount: float) -> str:
    """Draft (but do NOT issue) a refund. Use this to PROPOSE a refund for review.

    A draft is non-sensitive and safe. Issuing the actual money movement happens
    only after a human approves — the graph handles that, not you.
    """
    result = {
        "draft_id": f"DRAFT-{order_id}",
        "order_id": order_id,
        "amount": amount,
        "status": "drafted",
        "note": "Awaiting human approval before the refund is issued.",
    }
    _log("create_refund_draft", {"order_id": order_id, "amount": amount}, result)
    return json.dumps(result)


@tool
def log_to_crm(customer_id: str, note: str) -> str:
    """Attach a note to the customer's CRM record."""
    result = {"customer_id": customer_id, "note": note, "status": "logged"}
    _log("log_to_crm", {"customer_id": customer_id, "note": note}, result)
    return json.dumps(result)


# Tools exposed to the solver's ReAct loop. Note the deliberate absence of
# issue_refund and send_reply — those are graph-executed, not model-callable.
SOLVER_TOOLS = [get_order, get_tracking, check_refund_policy, create_refund_draft, log_to_crm]
SOLVER_TOOLS_BY_NAME = {t.name: t for t in SOLVER_TOOLS}


# --- Sensitive / deterministic tools executed by the graph --------------------

def issue_refund(order_id: str, amount: float, thread_id: str | None = None) -> dict:
    """SENSITIVE: actually issue a refund. Only the graph calls this, post-approval."""
    with db.get_conn() as conn:
        row = conn.execute("SELECT customer_id, amount FROM orders WHERE id = ?", (order_id,)).fetchone()
    customer_id = row["customer_id"] if row else None
    result = {
        "refund_id": f"RF-{order_id}",
        "order_id": order_id,
        "amount": amount,
        "status": "refunded",
    }
    db.log_action(
        "issue_refund", {"order_id": order_id, "amount": amount}, result,
        thread_id=thread_id, customer_id=customer_id,
    )
    return result


def send_reply(text: str, thread_id: str | None = None, customer_id: str | None = None) -> dict:
    """Deterministic 'send'. The respond node uses this to dispatch the final reply."""
    result = {"status": "sent", "chars": len(text)}
    db.log_action("send_reply", {"text": text}, result, thread_id=thread_id, customer_id=customer_id)
    return result
