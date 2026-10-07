"""CallVerse — one Streamlit app for support, approvals, and management.

This single app serves three roles, switched with the sidebar:

  👤 Customer  — chats with the agent. When the AI proposes a sensitive action
                 (a refund), the customer is told a specialist will review it and
                 the conversation ENDS for them; they are never blocked. Later,
                 once staff decide, the outcome shows up as a follow-up message.

  🧑‍💼 Staff    — sees the pending-approval QUEUE (the `approvals` table). Each item
                 shows what the AI proposed, with full context, and Approve/Reject
                 resumes that durable thread out-of-band via Command(resume=...).

  📊 Manager  — runs the existing CallVerse Digital Twin, compares a manual staffing
                 decision, and separately inspects selected Advisor/Quality interactions.

This works because LangGraph's interrupt() pauses the run to the SQLite checkpoint
(keyed by thread_id) and returns control right away, so the staff decision can
happen seconds or hours later, from this other view or process.

Run:  streamlit run app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

from callverse.dashboard.manager import render_manager
from helppilot import config, db, graph

st.set_page_config(page_title="CallVerse", page_icon="🛟", layout="wide")


# ------------------------------------------------------------- cached data ---

@st.cache_resource
def get_app():
    """One compiled graph (with its SQLite checkpointer) shared across reruns."""
    db.init_db()
    return graph.get_app()


@st.cache_data
def list_customers() -> list[tuple[str, str]]:
    with db.get_conn() as conn:
        return [(r["id"], r["name"]) for r in conn.execute("SELECT id, name FROM customers ORDER BY id")]


@st.cache_data
def customer_name(customer_id: str) -> str:
    with db.get_conn() as conn:
        row = conn.execute("SELECT name FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return row["name"] if row else customer_id


@st.cache_data
def orders_for(customer_id: str) -> list[dict]:
    with db.get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, item, amount, status FROM orders WHERE customer_id = ? ORDER BY id",
            (customer_id,))]


# Suggested prompt per order status, so testing is copy-paste.
_SUGGEST = {
    "lost": "my package never arrived, order {oid}",
    "in_transit": "where is my order {oid}?",
    "delivered": "I want a refund on {oid}, I changed my mind",
    "processing": "when will my order {oid} ship?",
}


def new_thread() -> None:
    """Start a fresh conversation thread for the customer view."""
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.awaiting = []      # threads parked in the staff queue
    st.session_state.last_state = {}


def actions_for_thread(thread_id: str) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT action, payload, created_at FROM actions_log WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------- state init ----

if "thread_id" not in st.session_state:
    new_thread()
if "role" not in st.session_state:
    st.session_state.role = "👤 Customer"

app = get_app()


# ============================================================= CUSTOMER =======

def check_for_updates() -> int:
    """Surface resolved refunds as follow-up messages (the async notification).

    For every thread we parked in the staff queue, see if it's no longer pending;
    if so, pull the final reply the graph persisted on resume and drop it into the
    chat as an assistant message. Returns how many updates were surfaced.
    """
    if not st.session_state.awaiting:
        return 0
    pending_threads = {p["thread_id"] for p in db.get_pending_approvals()}
    surfaced = 0
    still_waiting = []
    for item in st.session_state.awaiting:
        if item["thread_id"] in pending_threads:
            still_waiting.append(item)
            continue
        resp = db.latest_response_for_thread(item["thread_id"])
        if resp:
            st.session_state.messages.append({"role": "assistant", "content": "📬 " + resp["reply"]})
            surfaced += 1
        else:
            still_waiting.append(item)  # decided but not yet persisted; check again
    st.session_state.awaiting = still_waiting
    return surfaced


def render_customer(customer_id: str) -> None:
    st.title("💬 Customer Support Chat")
    if not config.GROQ_API_KEY:
        st.error(
            "Live Groq is unavailable because `GROQ_API_KEY` is not configured. "
            "The Manager Control Room and its offline Interaction Lab remain available."
        )
        return

    # Poll for any staff decisions that landed since the last interaction.
    check_for_updates()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.session_state.awaiting:
        n = len(st.session_state.awaiting)
        st.info(f"⏳ {n} request{'s' if n > 1 else ''} with our team. "
                "You can keep chatting — we'll post the outcome here.")
        if st.button("🔄 Check for updates"):
            st.rerun()

    if prompt := st.chat_input("Ask about an order, refund, or policy…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking… (triage → solver → review)"):
            try:
                res = graph.run_turn(prompt, customer_id, st.session_state.thread_id, app=app)
            except Exception as exc:  # noqa: BLE001 - provider failures must stay user-facing
                st.error(
                    "Live provider unavailable; no offline response was substituted. "
                    f"Error type: {type(exc).__name__}."
                )
                return
        st.session_state.last_state = res.get("state", {})

        if res["status"] == "interrupted":
            # Sensitive action → hand off to staff. Tell the customer it's being
            # handled (do NOT show them an approve button), park the thread in the
            # queue, and rotate to a fresh thread so they can keep chatting.
            info = res["interrupt"]
            ref = f"A-{info.get('approval_id')}"
            ack = (
                "✅ Thanks — I've looked into this and a specialist is reviewing your "
                f"request now (ref **{ref}**). You'll get an update here shortly; no need "
                "to wait."
            )
            st.session_state.messages.append({"role": "assistant", "content": ack})
            st.session_state.awaiting.append({
                "thread_id": st.session_state.thread_id,
                "ref": ref,
                "order_id": info.get("order_id"),
                "amount": info.get("amount"),
            })
            st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"  # rotate
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": res["state"].get("final_reply", "(no reply)")})
        st.rerun()


# ================================================================ STAFF ========

def render_staff() -> None:
    st.title("🧑‍💼 Staff — Approval Queue")
    st.caption("Sensitive actions the AI proposed. Approve to execute the refund; "
               "reject to decline. Each decision resumes a durable, paused agent run.")

    pending = db.get_pending_approvals()
    if not pending:
        st.success("🎉 Queue is clear — no approvals waiting.")
        if st.button("🔄 Refresh"):
            st.rerun()
        return

    st.markdown(f"**{len(pending)}** awaiting decision:")
    for p in pending:
        thread_id = p["thread_id"]
        payload = p["payload"]
        # Read what the AI proposed for this paused thread (no side effects).
        state = graph.peek_state(thread_id, app=app)
        cust_id = state.get("customer_id", "?")
        proposed = state.get("proposed_response", "")
        citations = state.get("citations", [])

        with st.container(border=True):
            st.markdown(
                f"### 💸 Refund ${payload.get('amount')} — order `{payload.get('order_id')}`")
            st.markdown(f"**Customer:** {customer_name(cust_id)} (`{cust_id}`) · "
                        f"ref **A-{p['id']}** · thread `{thread_id}`")
            if proposed:
                with st.expander("🤖 What the AI drafted for the customer", expanded=True):
                    st.markdown(proposed)
            if citations:
                st.caption("Grounded in: " + ", ".join(f"`{c}`" for c in citations))

            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("✅ Approve", key=f"ap-{p['id']}", type="primary"):
                with st.spinner("Issuing refund and finalizing reply…"):
                    graph.resume_turn(thread_id, approved=True, decided_by="staff", app=app)
                st.toast(f"Approved A-{p['id']} — refund issued.", icon="✅")
                st.rerun()
            if c2.button("❌ Reject", key=f"rj-{p['id']}"):
                with st.spinner("Declining and finalizing reply…"):
                    graph.resume_turn(thread_id, approved=False, decided_by="staff", app=app)
                st.toast(f"Rejected A-{p['id']}.", icon="❌")
                st.rerun()


# ------------------------------------------------------------- sidebar -------

with st.sidebar:
    st.header("🛟 CallVerse")
    st.caption("Digital Twin · HelpPilot · Quality Analyst")

    roles = ["👤 Customer", "🧑‍💼 Staff", "📊 Manager"]
    st.session_state.role = st.radio(
        "View as",
        roles,
        index=roles.index(st.session_state.role) if st.session_state.role in roles else 0,
        help="Customer support, staff approvals, and the manager control room share one app.",
    )
    is_customer = st.session_state.role == "👤 Customer"
    is_staff = st.session_state.role == "🧑‍💼 Staff"

    # A live badge so staff always know the queue depth.
    queue_depth = len(db.get_pending_approvals())
    if queue_depth:
        st.warning(f"🔔 {queue_depth} approval(s) in the staff queue")

    customer_id = None
    if is_customer:
        customers = list_customers()
        if not customers:
            st.warning("No demo customers are seeded. Run `python -m helppilot.seed`.")
        else:
            labels = [f"{cid} — {name}" for cid, name in customers]
            idx = st.selectbox("You are", range(len(customers)), format_func=lambda i: labels[i])
            new_id = customers[idx][0]
            # Switching customer starts a clean conversation.
            if st.session_state.get("customer_id") != new_id:
                st.session_state.customer_id = new_id
                new_thread()
            customer_id = new_id

            if st.button("🔄 New conversation", width="stretch"):
                new_thread()
                st.rerun()

            # Test-data cheat sheet: this customer's orders + copy-paste prompts.
            st.divider()
            st.subheader("🧪 This customer's orders")
            status_emoji = {"lost": "📦❌", "in_transit": "🚚", "delivered": "✅", "processing": "⏳"}
            for o in orders_for(customer_id):
                st.markdown(f"{status_emoji.get(o['status'], '•')} **{o['id']}** — {o['item']} "
                            f"· ${o['amount']:.2f} · `{o['status']}`")
                if _SUGGEST.get(o["status"]):
                    st.caption(f"try: “{_SUGGEST[o['status']].format(oid=o['id'])}”")

            with st.expander("📚 All test data (every customer & order)"):
                for cid, name in list_customers():
                    st.markdown(f"**{cid} — {name}**")
                    for o in orders_for(cid):
                        st.caption(f"{o['id']} · {o['item']} · ${o['amount']:.2f} · {o['status']}")

            # Trace panel: mirrors the LangSmith trace for the last turn.
            st.divider()
            st.subheader("🔎 Trace panel")
            state = st.session_state.get("last_state", {})
            if state:
                if state.get("route"):
                    st.markdown(f"**Triage route:** `{state['route']}`")
                    if state.get("triage_reason"):
                        st.caption(state["triage_reason"])
                for tc in state.get("tool_calls") or []:
                    st.markdown(f"- `{tc['name']}` {tc.get('args', {})}")
                if state.get("citations"):
                    st.markdown("**Citations:** " + ", ".join(f"`{c}`" for c in state["citations"]))
                review = state.get("review") or {}
                if review:
                    st.markdown(f"**Reviewer:** {'✅ approved' if review.get('approved') else '❌ rejected'}")

    st.divider()
    if config.setup_tracing():
        st.success("LangSmith tracing ON")
        st.markdown("[Open LangSmith ↗](https://smith.langchain.com/)")
    else:
        st.caption("LangSmith tracing off (set keys in .env)")


# ------------------------------------------------------------- main pane -----

if is_customer and customer_id is not None:
    render_customer(customer_id)
elif is_customer:
    st.title("💬 Customer Support Chat")
    st.info("Seed demo customers to start a support conversation.")
elif is_staff:
    render_staff()
else:
    render_manager()
