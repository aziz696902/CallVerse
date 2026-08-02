"""~30 test tickets with labels for the eval harness.

Each case carries the minimum ground truth an evaluator needs:
  expected_route     — for escalation-correctness.
  expect_refund      — whether the solver should PROPOSE a refund (drives the
                       "no refund without approval" hard check + resume path).
  criteria           — plain-language points a correct answer should hit
                       (LLM-as-judge reads these).
All refund scenarios use CUST-1001 / ORD-5001 (the carrier-lost order) unless a
case is deliberately testing a non-refundable situation.
"""
from __future__ import annotations

DATASET: list[dict] = [
    # --- Lost package → should propose refund (the core flow) ---
    {"id": "t01", "query": "My package never arrived and it's been two weeks. Order ORD-5001.",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "Confirms order ORD-5001 is lost, offers a full refund, cites refund policy."},
    {"id": "t02", "query": "Where is my order ORD-5001? Tracking hasn't moved.",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "Checks tracking, finds it lost, proposes a full refund with citation."},
    {"id": "t03", "query": "I think my headphones order got lost in the mail. ORD-5001.",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "Identifies lost package and offers refund per policy."},
    {"id": "t04", "query": "Carrier says my package ORD-5001 is lost. What now?",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "Acknowledges carrier-lost status, refund proposed, no return required."},

    # --- Order / tracking questions → tools, but NO refund ---
    {"id": "t05", "query": "Can you tell me the status of order ORD-5003?",
     "customer_id": "CUST-1003", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Reports ORD-5003 is in transit, arriving in about 2 days."},
    {"id": "t06", "query": "Was my order ORD-5002 delivered?",
     "customer_id": "CUST-1002", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Confirms ORD-5002 was delivered (left at front door)."},
    {"id": "t07", "query": "When will my standing desk ORD-5005 ship?",
     "customer_id": "CUST-1005", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Explains ORD-5005 is still processing / preparing for shipment."},
    {"id": "t08", "query": "What's the tracking number for ORD-5004?",
     "customer_id": "CUST-1004", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Provides tracking status for ORD-5004 (delivered, UPS)."},

    # --- FAQ / policy → answer_directly, no refund ---
    {"id": "t09", "query": "How long do refunds take to show up on my card?",
     "customer_id": "CUST-1002", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "States refunds take 5-7 business days to the original payment method."},
    {"id": "t10", "query": "What's your return window for unused items?",
     "customer_id": "CUST-1003", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "States 30-day return window for unused items."},
    {"id": "t11", "query": "How fast is express shipping?",
     "customer_id": "CUST-1004", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Express shipping is 1-2 business days."},
    {"id": "t12", "query": "Do you cover accidental damage under warranty?",
     "customer_id": "CUST-1004", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Explains warranty covers manufacturing defects, not accidental damage."},
    {"id": "t13", "query": "Can I cancel an order after it ships?",
     "customer_id": "CUST-1005", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Cannot cancel after shipping; refuse delivery or start a return."},
    {"id": "t14", "query": "What are your support hours?",
     "customer_id": "CUST-1001", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Support hours are 9am-6pm ET, Monday to Friday."},
    {"id": "t15", "query": "Can support agents see my full credit card number?",
     "customer_id": "CUST-1002", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Agents cannot view stored card details; full numbers are never stored."},
    {"id": "t16", "query": "How do I start a return?",
     "customer_id": "CUST-1003", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Start a return from the Orders page to get a prepaid label."},

    # --- Escalations → escalate_to_human ---
    {"id": "t17", "query": "This is unacceptable, I'm going to sue you and call my lawyer.",
     "customer_id": "CUST-1002", "expected_route": "escalate_to_human", "expect_refund": False,
     "criteria": "Recognizes a legal threat and escalates to a human."},
    {"id": "t18", "query": "Your product started smoking and nearly caused a fire!",
     "customer_id": "CUST-1004", "expected_route": "escalate_to_human", "expect_refund": False,
     "criteria": "Treats this as a safety issue and escalates to a human."},
    {"id": "t19", "query": "I'm disputing three charges with my bank and want a full audit.",
     "customer_id": "CUST-1005", "expected_route": "escalate_to_human", "expect_refund": False,
     "criteria": "Complex billing dispute is escalated to a specialist."},
    {"id": "t20", "query": "I need to speak to a human manager right now about a legal matter.",
     "customer_id": "CUST-1001", "expected_route": "escalate_to_human", "expect_refund": False,
     "criteria": "Escalates to a human specialist."},

    # --- More lost / delivery-problem refunds ---
    {"id": "t21", "query": "It's been 15 days with no tracking update on ORD-5001. Refund please.",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "10+ days without a scan qualifies for a full refund; cites policy."},
    {"id": "t22", "query": "My order ORD-5001 shows lost. I want my money back.",
     "customer_id": "CUST-1001", "expected_route": "use_tools", "expect_refund": True,
     "criteria": "Confirms lost, proposes full refund of the order amount."},

    # --- Refund NOT warranted (delivered) → tools, no refund ---
    {"id": "t23", "query": "I want a refund on ORD-5002, I just changed my mind.",
     "customer_id": "CUST-1002", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Order was delivered; explains return process rather than auto-refunding a lost package."},
    {"id": "t24", "query": "Refund my ORD-5004 earbuds, they were delivered but I don't want them.",
     "customer_id": "CUST-1004", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Delivered item — points to 30-day return, not a lost-package refund."},

    # --- Mixed FAQ ---
    {"id": "t25", "query": "Are gift cards returnable?",
     "customer_id": "CUST-1003", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Gift cards and final-sale items cannot be returned."},
    {"id": "t26", "query": "When does an order count as officially lost?",
     "customer_id": "CUST-1001", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Lost after 10+ days without a scan or a carrier lost mark."},
    {"id": "t27", "query": "How long is the warranty on electronics?",
     "customer_id": "CUST-1004", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "1-year limited warranty on electronics."},
    {"id": "t28", "query": "Do you offer exchanges?",
     "customer_id": "CUST-1003", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Exchanges available within 30 days for unused items."},
    {"id": "t29", "query": "What payment methods do you accept?",
     "customer_id": "CUST-1002", "expected_route": "answer_directly", "expect_refund": False,
     "criteria": "Major credit cards and store credit."},
    {"id": "t30", "query": "My tracking says delivered but I never got ORD-5002. Help!",
     "customer_id": "CUST-1002", "expected_route": "use_tools", "expect_refund": False,
     "criteria": "Advises checking neighbors and waiting 48 hours before treating as lost."},
]
