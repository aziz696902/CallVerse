from __future__ import annotations

from dataclasses import dataclass

import pytest

from callverse.classification import ClassificationResult
from callverse.customer_advisor import CallVerseCustomerAdvisor, RoutingPath
from callverse.domain import RequestIntent, SupportRequest, Urgency
from helppilot.kb_docs import KB_DOCS
from helppilot.seed import ORDERS


@dataclass
class FakeClassifier:
    result: ClassificationResult | Exception

    def classify(self, _text: str) -> ClassificationResult:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def classification(
    intent: RequestIntent,
    confidence: float = 0.95,
    needs_review: bool = False,
    order_id: str | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        needs_review=needs_review,
        extracted_order_id=order_id,
        urgency=Urgency.NORMAL,
        model_name="test-model",
        model_version="test-v1",
    )


def request(message: str, *, customer_id: str = "CUST-1003") -> SupportRequest:
    return SupportRequest(
        request_id="REQ-TEST",
        customer_id=customer_id,
        intent=RequestIntent.GENERAL,
        customer_message=message,
        arrival_time=0,
    )


def found_order(order_id: str, customer_id: str, _request_id: str) -> dict[str, str]:
    return {"id": order_id, "customer_id": customer_id, "status": "in_transit"}


def done_runner(calls: list[str]):
    def run(message: str, _customer_id: str, _request_id: str):
        calls.append(message)
        return {
            "status": "done",
            "state": {
                "final_reply": "Verified response",
                "route": "use_tools",
                "tool_calls": [{"name": "get_tracking"}],
                "citations": ["shipping-delivery"],
            },
        }

    return run


def test_high_confidence_tracking_uses_structured_classifier_route():
    calls: list[str] = []
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.TRACKING, order_id="ORD-5003")),
        runner=done_runner(calls),
        order_lookup=found_order,
    )

    interaction = advisor.handle_with_trace(request("Where is ORD-5003?"))

    assert "accepted_intent: tracking" in calls[0]
    assert "order_id: ORD-5003" in calls[0]
    assert interaction.metadata.routing_path is RoutingPath.CLASSIFIER
    assert interaction.metadata.classifier_accepted is True
    assert interaction.metadata.tools_used == ("get_order", "get_tracking")
    assert interaction.metadata.rag_sources == ("shipping-delivery",)
    assert interaction.result.resolved is True


def test_high_confidence_refund_routes_correctly():
    calls: list[str] = []
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.REFUND)), runner=done_runner(calls)
    )

    interaction = advisor.handle_with_trace(request("I need a refund"))

    assert "accepted_intent: refund" in calls[0]
    assert interaction.metadata.predicted_intent is RequestIntent.REFUND
    assert interaction.metadata.classifier_accepted is True


def test_low_confidence_uses_unchanged_helppilot_fallback():
    calls: list[str] = []
    message = "Could you maybe sort this out?"
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.COMPLAINT, 0.31, True)),
        runner=done_runner(calls),
    )

    interaction = advisor.handle_with_trace(request(message))

    assert calls == [message]
    assert interaction.metadata.routing_path is RoutingPath.HELPPILOT_FALLBACK
    assert interaction.metadata.fallback_reason == "low_confidence"


def test_damaged_item_language_forces_unsupported_fallback():
    calls: list[str] = []
    message = "My item arrived broken"
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.COMPLAINT)), runner=done_runner(calls)
    )

    interaction = advisor.handle_with_trace(request(message))

    assert calls == [message]
    assert interaction.metadata.fallback_reason == "unsupported_damaged_item"
    assert interaction.metadata.classifier_accepted is False


def test_general_ambiguous_language_uses_fallback():
    calls: list[str] = []
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.TRACKING)), runner=done_runner(calls)
    )

    interaction = advisor.handle_with_trace(request("Hello!"))

    assert calls == ["Hello!"]
    assert interaction.metadata.fallback_reason == "general_or_ambiguous"


def test_order_id_is_passed_to_lookup_and_context():
    lookups: list[tuple[str, str, str]] = []
    calls: list[str] = []

    def lookup(order_id: str, customer_id: str, request_id: str):
        lookups.append((order_id, customer_id, request_id))
        return found_order(order_id, customer_id, request_id)

    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.CANCEL_ORDER, order_id="ORD-5005")),
        runner=done_runner(calls),
        order_lookup=lookup,
    )
    advisor.handle_with_trace(request("Cancel ORD-5005", customer_id="CUST-1005"))

    assert lookups == [("ORD-5005", "CUST-1005", "REQ-TEST")]
    assert "order_id: ORD-5005" in calls[0]


def test_nonexistent_order_never_calls_runner_or_invents_status():
    def runner(*_args):
        pytest.fail("HelpPilot must not run after a failed order preflight")

    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.TRACKING, order_id="ORD-9999")),
        runner=runner,
        order_lookup=lambda *_: {"error": "not found"},
    )

    interaction = advisor.handle_with_trace(request("Where is ORD-9999?"))

    assert interaction.metadata.routing_path is RoutingPath.SAFE_ORDER_ESCALATION
    assert interaction.result.escalated is True
    assert interaction.result.escalation_reason == "order_not_found"
    assert "guessing" in interaction.result.response_text
    assert "delivered" not in interaction.result.response_text


def test_helppilot_approval_interrupt_maps_to_escalation():
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.REFUND)),
        runner=lambda *_: {"status": "interrupted", "state": {"route": "use_tools"}},
    )

    interaction = advisor.handle_with_trace(request("Refund this purchase"))

    assert interaction.result.escalated is True
    assert interaction.result.escalation_reason == "human_approval_required"
    assert interaction.metadata.helppilot_status == "interrupted"


def test_classifier_failure_falls_back_safely():
    calls: list[str] = []
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(RuntimeError("inference failed")), runner=done_runner(calls)
    )

    interaction = advisor.handle_with_trace(request("Please help with billing"))

    assert calls == ["Please help with billing"]
    assert interaction.metadata.fallback_reason == "classifier_failure"


def test_missing_model_artifact_uses_documented_fallback(tmp_path):
    calls: list[str] = []
    advisor = CallVerseCustomerAdvisor.from_local_artifact(
        tmp_path / "missing.json", runner=done_runner(calls)
    )

    interaction = advisor.handle_with_trace(request("Please help with billing"))

    assert calls == ["Please help with billing"]
    assert interaction.metadata.fallback_reason == "classifier_unavailable"


def test_order_customer_mismatch_escalates_without_runner():
    advisor = CallVerseCustomerAdvisor(
        FakeClassifier(classification(RequestIntent.TRACKING, order_id="ORD-5003")),
        runner=lambda *_: pytest.fail("runner should not be called"),
        order_lookup=lambda *_: {"id": "ORD-5003", "customer_id": "CUST-OTHER"},
    )

    interaction = advisor.handle_with_trace(request("Track ORD-5003"))

    assert interaction.result.escalation_reason == "order_customer_mismatch"
    assert interaction.metadata.tools_used == ("get_order",)


def test_delivery_demo_seed_remains_small_and_covers_required_order_states():
    statuses = {row[4] for row in ORDERS}

    assert len(ORDERS) == 6
    assert {"in_transit", "delayed", "lost", "processing", "delivered"} <= statuses


def test_callverse_policy_additions_are_explicitly_synthetic():
    additions = [doc for doc in KB_DOCS if doc["id"].startswith("callverse-")]

    assert {doc["id"] for doc in additions} == {
        "callverse-address-change-demo",
        "callverse-damaged-item-demo",
    }
    assert all("synthetic demonstration policy" in doc["text"].lower() for doc in additions)
