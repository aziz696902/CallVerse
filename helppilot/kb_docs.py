"""The knowledge base: ~8 policy/FAQ documents.

Kept as plain Python so both the seed script (which embeds them into Chroma) and
tests can import the exact same source text. Each doc has a stable `id` used as
the citation handle the agent surfaces to the user (e.g. "[refund-policy]").
"""
from __future__ import annotations

KB_DOCS: list[dict[str, str]] = [
    {
        "id": "refund-policy",
        "title": "Refund Policy",
        "text": (
            "Refunds are available within 30 days of the delivery date for items in "
            "original condition. Refunds are issued to the original payment method and "
            "take 5-7 business days to appear. "
            "Lost packages: if a carrier marks an order as lost, or an order shows no "
            "tracking movement for 10 or more days, the customer is entitled to a full "
            "refund of the order amount, including original shipping. No return is "
            "required for lost packages. "
            "Refunds above the order amount are never permitted. Partial refunds may be "
            "offered for items damaged in transit at the agent's discretion."
        ),
    },
    {
        "id": "shipping-delivery",
        "title": "Shipping & Delivery",
        "text": (
            "Standard shipping takes 3-5 business days; express takes 1-2 business days. "
            "Orders placed before 2pm ET ship the same business day. Tracking numbers are "
            "emailed once the carrier scans the package. Delivery estimates are not "
            "guarantees; carrier delays do not by themselves qualify an order as lost "
            "until 10 days without a scan have elapsed."
        ),
    },
    {
        "id": "lost-missing-packages",
        "title": "Lost or Missing Packages",
        "text": (
            "If tracking shows 'delivered' but the customer did not receive the package, "
            "advise checking with neighbors and the building office, then wait 48 hours as "
            "packages are sometimes scanned early. If tracking has not updated for 10 or "
            "more days, or the carrier explicitly marks the order 'lost', treat the "
            "package as lost and process a full refund per the Refund Policy. Always "
            "confirm the carrier status with get_tracking before deciding."
        ),
    },
    {
        "id": "returns-exchanges",
        "title": "Returns & Exchanges",
        "text": (
            "Customers may return unused items within 30 days for a full refund or "
            "exchange. Start a return from the Orders page to receive a prepaid label. "
            "Final-sale items and gift cards cannot be returned. Exchanges ship once the "
            "original item is scanned by the return carrier."
        ),
    },
    {
        "id": "order-cancellation",
        "title": "Order Cancellation",
        "text": (
            "Orders can be cancelled for a full refund any time before they ship. Once an "
            "order has shipped it cannot be cancelled; the customer should instead refuse "
            "delivery or start a return. Cancellations are processed immediately and the "
            "hold on the payment method is released within 1-3 business days."
        ),
    },
    {
        "id": "warranty",
        "title": "Warranty",
        "text": (
            "Electronics carry a 1-year limited warranty covering manufacturing defects. "
            "Accidental damage and normal wear are not covered. Warranty claims require "
            "the order number and a description of the defect and are handled as a "
            "replacement, not a cash refund."
        ),
    },
    {
        "id": "account-payment",
        "title": "Account & Payment",
        "text": (
            "We accept major credit cards and store credit. A temporary authorization hold "
            "may appear at checkout and clears in 1-3 business days. We never store full "
            "card numbers. Customers can update their payment method from Account Settings; "
            "support agents cannot view or change stored card details."
        ),
    },
    {
        "id": "contact-escalation",
        "title": "Contact & Escalation",
        "text": (
            "Support hours are 9am-6pm ET, Monday to Friday. Complex billing disputes, "
            "legal threats, safety issues, or any request the agent is not authorized to "
            "resolve must be escalated to a human specialist. Escalations create a ticket "
            "and promise a response within one business day."
        ),
    },
]
