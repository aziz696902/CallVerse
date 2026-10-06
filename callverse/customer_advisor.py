"""Classifier-first orchestration over the inherited HelpPilot advisor."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

from .advisor import HelpPilotRunner, map_helppilot_result
from .classification import ClassificationResult, RequestClassifier, extract_order_id
from .domain import (
    AdvisorResult,
    CustomerProfile,
    DomainModel,
    RequestIntent,
    SupportRequest,
)

SUPPORTED_INTENTS = frozenset(
    {
        RequestIntent.TRACKING,
        RequestIntent.REFUND,
        RequestIntent.CANCEL_ORDER,
        RequestIntent.ADDRESS_CHANGE,
        RequestIntent.PAYMENT_ISSUE,
        RequestIntent.COMPLAINT,
    }
)

_DAMAGED_LANGUAGE = re.compile(
    r"\b(damaged|broken|cracked|smashed|defective|shattered|torn)\b", re.IGNORECASE
)
_GENERAL_LANGUAGE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|help|good (morning|afternoon|evening))[!.?\s]*$",
    re.IGNORECASE,
)


class RoutingPath(str, Enum):
    CLASSIFIER = "classifier"
    HELPPILOT_FALLBACK = "helppilot_fallback"
    SAFE_ORDER_ESCALATION = "safe_order_escalation"


class InteractionMetadata(DomainModel):
    """Compact observable trace; it deliberately contains no model reasoning."""

    routing_path: RoutingPath
    predicted_intent: RequestIntent | None = None
    classifier_confidence: float | None = None
    classifier_accepted: bool = False
    fallback_reason: str | None = None
    extracted_order_id: str | None = None
    urgency: str | None = None
    helppilot_status: str | None = None
    helppilot_route: str | None = None
    tools_used: tuple[str, ...] = ()
    rag_sources: tuple[str, ...] = ()
    resolved: bool
    escalated: bool


class AdvisorInteraction(DomainModel):
    result: AdvisorResult
    metadata: InteractionMetadata


OrderLookup = Callable[[str, str, str], dict[str, Any]]


class CallVerseCustomerAdvisor:
    """Coordinate classification and the existing HelpPilot graph.

    Accepted predictions are supplied as explicit routing context. Uncertain,
    unsupported, or unavailable predictions preserve HelpPilot's normal triage.
    """

    def __init__(
        self,
        classifier: RequestClassifier | None,
        runner: HelpPilotRunner | None = None,
        order_lookup: OrderLookup | None = None,
        classifier_unavailable_reason: str | None = None,
    ) -> None:
        self._classifier = classifier
        self._runner = runner or self._default_runner
        self._order_lookup = order_lookup or self._default_order_lookup
        self._classifier_unavailable_reason = classifier_unavailable_reason
        self.last_interaction: AdvisorInteraction | None = None

    @classmethod
    def from_local_artifact(
        cls,
        metadata_path: str | Path = "data/processed/classification/evaluation/model_metadata.json",
        **kwargs: Any,
    ) -> CallVerseCustomerAdvisor:
        try:
            classifier = RequestClassifier.from_metadata(metadata_path)
        except Exception as exc:  # noqa: BLE001 - any load failure must degrade safely
            return cls(
                classifier=None,
                classifier_unavailable_reason=f"{type(exc).__name__}: {exc}",
                **kwargs,
            )
        return cls(classifier=classifier, **kwargs)

    @staticmethod
    def _default_runner(message: str, customer_id: str, request_id: str) -> dict[str, Any]:
        from helppilot.graph import run_turn

        return run_turn(message, customer_id, request_id)

    @staticmethod
    def _default_order_lookup(order_id: str, customer_id: str, request_id: str) -> dict[str, Any]:
        from helppilot.tools import get_order, set_run_context

        set_run_context(request_id, customer_id)
        return json.loads(get_order.invoke({"order_id": order_id}))

    @staticmethod
    def _fallback_reason(message: str, classification: ClassificationResult) -> str | None:
        if _DAMAGED_LANGUAGE.search(message):
            return "unsupported_damaged_item"
        if _GENERAL_LANGUAGE.fullmatch(message):
            return "general_or_ambiguous"
        if classification.intent not in SUPPORTED_INTENTS:
            return "unsupported_intent"
        if classification.needs_review:
            return "low_confidence"
        return None

    @staticmethod
    def _structured_message(message: str, classification: ClassificationResult, order_id: str | None) -> str:
        return (
            "[CALLVERSE ROUTING CONTEXT]\n"
            f"accepted_intent: {classification.intent.value}\n"
            f"order_id: {order_id or 'not_provided'}\n"
            f"urgency: {classification.urgency.value}\n"
            "instruction: Treat this as routing metadata. Verify customer-specific facts "
            "with business tools and never infer an order status.\n"
            "[CUSTOMER MESSAGE]\n"
            f"{message}"
        )

    @staticmethod
    def _execution_metadata(result: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
        state = result.get("state") or {}
        tool_names: list[str] = []
        for call in state.get("tool_calls") or ():
            if isinstance(call, dict):
                name = call.get("name")
            else:
                name = getattr(call, "name", None)
            if name:
                tool_names.append(str(name))
        citations = tuple(str(item) for item in (state.get("citations") or ()))
        return tuple(tool_names), citations, state.get("route")

    def _safe_order_result(
        self,
        request: SupportRequest,
        reason: str,
        predicted: ClassificationResult | None,
        order_id: str,
    ) -> AdvisorInteraction:
        result = AdvisorResult(
            request_id=request.request_id,
            resolved=False,
            escalated=True,
            automated=False,
            response_text=(
                "I could not verify that order for this customer. I have sent the request "
                "for human review rather than guessing its status."
            ),
            escalation_reason=reason,
            handling_duration=0.0,
        )
        return AdvisorInteraction(
            result=result,
            metadata=InteractionMetadata(
                routing_path=RoutingPath.SAFE_ORDER_ESCALATION,
                predicted_intent=predicted.intent if predicted else None,
                classifier_confidence=predicted.confidence if predicted else None,
                classifier_accepted=False,
                fallback_reason=reason,
                extracted_order_id=order_id,
                urgency=predicted.urgency.value if predicted else None,
                tools_used=("get_order",),
                resolved=False,
                escalated=True,
            ),
        )

    def handle_with_trace(
        self,
        request: SupportRequest,
        customer: CustomerProfile | None = None,
    ) -> AdvisorInteraction:
        if customer is not None and customer.customer_id != request.customer_id:
            raise ValueError("customer profile ID must match the support request customer ID")

        classification: ClassificationResult | None = None
        fallback_reason: str | None = None
        try:
            if self._classifier is None:
                fallback_reason = "classifier_unavailable"
            else:
                classification = self._classifier.classify(request.customer_message)
                fallback_reason = self._fallback_reason(request.customer_message, classification)
        except Exception:  # noqa: BLE001 - inference providers may raise arbitrary errors
            fallback_reason = "classifier_failure"

        order_id = (
            classification.extracted_order_id if classification else None
        ) or extract_order_id(request.customer_message) or request.order_id

        if order_id:
            try:
                order = self._order_lookup(order_id, request.customer_id, request.request_id)
            except Exception:  # noqa: BLE001 - business lookup failure must not leak facts
                interaction = self._safe_order_result(
                    request, "order_lookup_failure", classification, order_id
                )
                self.last_interaction = interaction
                return interaction
            if order.get("error"):
                interaction = self._safe_order_result(request, "order_not_found", classification, order_id)
                self.last_interaction = interaction
                return interaction
            if str(order.get("customer_id")) != request.customer_id:
                interaction = self._safe_order_result(
                    request, "order_customer_mismatch", classification, order_id
                )
                self.last_interaction = interaction
                return interaction

        accepted = classification is not None and fallback_reason is None
        routed_message = (
            self._structured_message(request.customer_message, classification, order_id)
            if accepted and classification is not None
            else request.customer_message
        )
        started = perf_counter()
        raw = self._runner(routed_message, request.customer_id, request.request_id)
        elapsed = perf_counter() - started
        advisor_result = map_helppilot_result(request.request_id, raw, elapsed)
        tools_used, citations, helppilot_route = self._execution_metadata(raw)
        if order_id:
            tools_used = tuple(dict.fromkeys(("get_order", *tools_used)))
        interaction = AdvisorInteraction(
            result=advisor_result,
            metadata=InteractionMetadata(
                routing_path=(RoutingPath.CLASSIFIER if accepted else RoutingPath.HELPPILOT_FALLBACK),
                predicted_intent=classification.intent if classification else None,
                classifier_confidence=classification.confidence if classification else None,
                classifier_accepted=accepted,
                fallback_reason=fallback_reason,
                extracted_order_id=order_id,
                urgency=classification.urgency.value if classification else None,
                helppilot_status=str(raw.get("status")) if raw.get("status") is not None else None,
                helppilot_route=str(helppilot_route) if helppilot_route is not None else None,
                tools_used=tools_used,
                rag_sources=citations,
                resolved=advisor_result.resolved,
                escalated=advisor_result.escalated,
            ),
        )
        self.last_interaction = interaction
        return interaction

    def handle_support_request(
        self,
        request: SupportRequest,
        customer: CustomerProfile | None = None,
    ) -> AdvisorResult:
        return self.handle_with_trace(request, customer).result
