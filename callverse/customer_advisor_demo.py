"""Deterministic, network-free demonstration of CallVerse advisor routing."""

from __future__ import annotations

import argparse
import json

from .classification import extract_order_id
from .customer_advisor import CallVerseCustomerAdvisor
from .domain import RequestIntent, SupportRequest


def deterministic_demo_runner(message: str, customer_id: str, request_id: str) -> dict:
    from helppilot.tools import check_refund_policy, get_tracking, set_run_context

    set_run_context(request_id, customer_id)
    accepted = "accepted_intent:" in message
    intent = message.split("accepted_intent: ", 1)[1].splitlines()[0] if accepted else "fallback"
    tool_calls: list[dict[str, str]] = []
    citations: list[str] = []
    reply = "Deterministic fallback reached; no live LLM response was generated."
    if intent == "tracking":
        order_id = extract_order_id(message)
        tracking = json.loads(get_tracking.invoke({"order_id": order_id}))
        tool_calls.append({"name": "get_tracking"})
        if tracking.get("error"):
            reply = tracking["error"]
        else:
            reply = f"{tracking['order_id']} is {tracking['status']}. Last scan: {tracking['last_scan']}"
    elif intent == "refund":
        policy = json.loads(check_refund_policy.invoke({"query": message}))
        tool_calls.append({"name": "check_refund_policy"})
        citations = [item["doc_id"] for item in policy["results"]]
        reply = "Refund policy retrieved for deterministic review; no refund was issued."
    requires_approval = intent == "refund"
    fallback = not accepted
    return {
        "status": "interrupted" if requires_approval else "done",
        "state": {
            "final_reply": reply,
            "route": "use_tools" if accepted else "inherited_triage",
            "tool_calls": tool_calls,
            "citations": citations,
            "escalated": fallback,
            "triage_reason": "offline_demo_requires_live_triage" if fallback else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?", default="Where is ORD-5003?")
    parser.add_argument("--customer-id", default="CUST-1003")
    args = parser.parse_args()
    advisor = CallVerseCustomerAdvisor.from_local_artifact(runner=deterministic_demo_runner)
    request = SupportRequest(
        request_id="DEMO-001",
        customer_id=args.customer_id,
        intent=RequestIntent.GENERAL,
        customer_message=args.message,
        arrival_time=0,
    )
    interaction = advisor.handle_with_trace(request)
    print("DETERMINISTIC OFFLINE DEMO (no live LLM)")
    print(json.dumps(interaction.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
