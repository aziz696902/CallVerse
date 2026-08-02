"""SQLite schema + tiny data-access helpers.

One physical database (config.DB_PATH) holds all the *business* data the agent
reasons over and writes to: customers, their tickets, per-customer stored facts
(a crude long-term memory), orders, plus two audit-style tables the agent writes
during a run — actions_log (everything it did) and approvals (the human gate for
sensitive actions). LangGraph's own durable checkpoint state lives in a separate
file (config.CHECKPOINT_DB_PATH) and is managed by the checkpointer, not here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    subject     TEXT NOT NULL,
    status      TEXT NOT NULL,        -- open | resolved | escalated
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stored_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    fact        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    item            TEXT NOT NULL,
    amount          REAL NOT NULL,
    status          TEXT NOT NULL,    -- delivered | in_transit | lost | processing
    carrier         TEXT,
    tracking_number TEXT,
    last_scan       TEXT,             -- human-readable last carrier scan
    created_at      TEXT NOT NULL
);

-- Audit trail: every tool action the agent takes is appended here.
CREATE TABLE IF NOT EXISTS actions_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT,
    customer_id TEXT,
    action      TEXT NOT NULL,
    payload     TEXT,                 -- JSON
    result      TEXT,                 -- JSON
    created_at  TEXT NOT NULL
);

-- Human-in-the-loop gate for sensitive actions (e.g. issue_refund).
CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     TEXT,                 -- JSON
    status      TEXT NOT NULL,        -- pending | approved | rejected
    decided_by  TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT
);

-- Final replies the agent sends, persisted by the respond node.
CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    customer_id TEXT,
    ticket_id   TEXT,
    reply       TEXT NOT NULL,
    citations   TEXT,                 -- JSON list
    created_at  TEXT NOT NULL
);
"""


def now() -> str:
    """UTC timestamp as an ISO-8601 string (what every created_at column stores)."""
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    """Open a connection with row access by name and FK enforcement on."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist (idempotent)."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- Small write helpers used by tools / nodes --------------------------------

def log_action(
    action: str,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    thread_id: str | None = None,
    customer_id: str | None = None,
) -> None:
    """Append one row to actions_log. Called by every tool for the audit trail."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO actions_log (thread_id, customer_id, action, payload, result, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                customer_id,
                action,
                json.dumps(payload) if payload is not None else None,
                json.dumps(result) if result is not None else None,
                now(),
            ),
        )


def record_approval(thread_id: str, action: str, payload: dict[str, Any]) -> int:
    """Get-or-create a pending approval row for this thread; return its id.

    IMPORTANT: LangGraph re-executes the code *before* an interrupt() when a paused
    run resumes. If this blindly INSERTed, each resume would leave a duplicate,
    forever-'pending' ghost row in the staff queue. So we reuse an existing pending
    row for the same (thread_id, action) and only insert when none exists.
    """
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM approvals WHERE thread_id = ? AND action = ? AND status = 'pending'"
            " ORDER BY id DESC LIMIT 1",
            (thread_id, action),
        ).fetchone()
        if existing:
            return int(existing["id"])
        cur = conn.execute(
            "INSERT INTO approvals (thread_id, action, payload, status, created_at)"
            " VALUES (?, ?, ?, 'pending', ?)",
            (thread_id, action, json.dumps(payload), now()),
        )
        return int(cur.lastrowid)


def get_pending_approvals() -> list[dict[str, Any]]:
    """All approvals awaiting a staff decision — the admin dashboard's work queue."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, action, payload, created_at FROM approvals"
            " WHERE status = 'pending' ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
        out.append(d)
    return out


def latest_response_for_thread(thread_id: str) -> dict[str, Any] | None:
    """The most recent persisted reply for a thread (used to notify the customer)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT reply, citations, created_at FROM responses WHERE thread_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def decide_approval(approval_id: int, approved: bool, decided_by: str = "human") -> None:
    """Mark an approval approved/rejected (mirrors the LangGraph resume decision)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ? WHERE id = ?",
            ("approved" if approved else "rejected", decided_by, now(), approval_id),
        )


def save_response(
    thread_id: str,
    reply: str,
    citations: list[str],
    customer_id: str | None = None,
    ticket_id: str | None = None,
) -> None:
    """Persist the final outgoing reply (called by the respond node)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO responses (thread_id, customer_id, ticket_id, reply, citations, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, customer_id, ticket_id, reply, json.dumps(citations), now()),
        )
