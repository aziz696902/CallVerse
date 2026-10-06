"""Idempotent seed script: business data (SQLite) + knowledge base (Chroma).

Run as `python -m helppilot.seed` or via the console. Re-running wipes and
reinserts only the seed-owned tables (customers, tickets, stored_facts, orders)
and rebuilds the Chroma collection, so the demo always starts from a known state.
Runtime tables (actions_log, approvals, responses) are left untouched.

If chromadb/sentence-transformers are not installed, seeding the SQLite half still
succeeds and prints a note instead of failing.
"""
from __future__ import annotations

from . import config, db

# --- Seed content -------------------------------------------------------------

CUSTOMERS = [
    # id, name, email
    ("CUST-1001", "Alice Nguyen", "alice@example.com"),
    ("CUST-1002", "Bob Martinez", "bob@example.com"),
    ("CUST-1003", "Carla Singh", "carla@example.com"),
    ("CUST-1004", "David Okoye", "david@example.com"),
    ("CUST-1005", "Emma Larsson", "emma@example.com"),
]

# customer_id -> list of (fact,)
STORED_FACTS = {
    "CUST-1001": ["Prefers email contact.", "VIP: 3 years, 20+ orders."],
    "CUST-1002": ["Had a delayed shipment refunded in 2025."],
    "CUST-1003": ["Requested paperless receipts."],
    "CUST-1004": ["Prefers express shipping."],
    "CUST-1005": ["Business account; bulk orders."],
}

# id, customer_id, subject, status
TICKETS = [
    ("TCK-9001", "CUST-1001", "Where is my order?", "resolved"),
    ("TCK-9002", "CUST-1002", "Refund for late delivery", "resolved"),
    ("TCK-9003", "CUST-1003", "How do I return an item?", "resolved"),
    ("TCK-9004", "CUST-1004", "Warranty claim on headphones", "open"),
    ("TCK-9005", "CUST-1005", "Bulk order invoice question", "open"),
]

# id, customer_id, item, amount, status, carrier, tracking_number, last_scan
ORDERS = [
    # The lost order that drives the refund flow.
    ("ORD-5001", "CUST-1001", "Wireless Headphones", 129.99, "lost",
     "UPS", "1Z999AA10123456784", "Lost in transit — reported by carrier"),
    ("ORD-5002", "CUST-1002", "Mechanical Keyboard", 89.00, "delivered",
     "FedEx", "7712 3456 7890", "Delivered, left at front door"),
    ("ORD-5003", "CUST-1003", "USB-C Cable (2-pack)", 15.50, "in_transit",
     "USPS", "9400 1000 0000 0000 0000 00", "In transit, arriving in 2 days"),
    ("ORD-5004", "CUST-1004", "Noise-Cancelling Earbuds", 199.99, "delivered",
     "UPS", "1Z999AA10987654321", "Delivered, signed by D. Okoye"),
    ("ORD-5005", "CUST-1005", "Standing Desk", 349.00, "processing",
     None, None, "Preparing for shipment"),
    ("ORD-5006", "CUST-1002", "Laptop Stand", 54.00, "delayed",
     "DHL", "JD014600006281234567", "Carrier delay; estimated arrival in 3 days"),
]


def seed_sqlite() -> None:
    """Wipe and reinsert the seed-owned tables (idempotent)."""
    db.init_db()
    with db.get_conn() as conn:
        # Delete children before parents to satisfy foreign keys.
        for table in ("stored_facts", "tickets", "orders", "customers"):
            conn.execute(f"DELETE FROM {table}")

        conn.executemany(
            "INSERT INTO customers (id, name, email, created_at) VALUES (?, ?, ?, ?)",
            [(cid, name, email, db.now()) for cid, name, email in CUSTOMERS],
        )
        conn.executemany(
            "INSERT INTO tickets (id, customer_id, subject, status, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [(tid, cid, subj, status, db.now()) for tid, cid, subj, status in TICKETS],
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, item, amount, status, carrier,"
            " tracking_number, last_scan, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*row, db.now()) for row in ORDERS],
        )
        facts = [
            (cid, fact, db.now())
            for cid, fact_list in STORED_FACTS.items()
            for fact in fact_list
        ]
        conn.executemany(
            "INSERT INTO stored_facts (customer_id, fact, created_at) VALUES (?, ?, ?)",
            facts,
        )
    print(
        f"  SQLite: {len(CUSTOMERS)} customers, {len(TICKETS)} tickets, "
        f"{len(ORDERS)} orders (1 lost, 1 delayed), "
        f"{sum(len(f) for f in STORED_FACTS.values())} facts."
    )


def seed_chroma() -> None:
    """Embed the KB docs into a persistent Chroma collection."""
    try:
        from .rag import build_index
    except Exception as exc:  # pragma: no cover - only if RAG deps are missing
        print(f"  Chroma: skipped (RAG not available: {exc}).")
        return
    n = build_index()
    print(f"  Chroma: indexed {n} chunks into collection '{config.CHROMA_COLLECTION}'.")


def main() -> None:
    print("Seeding HelpPilot...")
    seed_sqlite()
    seed_chroma()
    print("Done.")


if __name__ == "__main__":
    main()
