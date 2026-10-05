"""Advisor boundary and a thin adapter for the inherited HelpPilot graph."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from .domain import AdvisorResult, CustomerProfile, SupportRequest

HelpPilotRunner = Callable[[str, str, str], dict[str, Any]]


@runtime_checkable
class Advisor(Protocol):
    """Interface used by a future simulator, independent of advisor internals."""

    def handle_support_request(
        self,
        request: SupportRequest,
        customer: CustomerProfile | None = None,
    ) -> AdvisorResult:
        ...


def map_helppilot_result(
    request_id: str,
    result: dict[str, Any],
    handling_duration: float | None = None,
) -> AdvisorResult:
    """Deterministically map HelpPilot's public run result to the domain contract."""

    state = result.get("state") or {}
    interrupted = result.get("status") == "interrupted"
    escalated = interrupted or bool(state.get("escalated", False))

    if interrupted:
        response_text = "This request requires human approval before completion."
        escalation_reason = "human_approval_required"
    else:
        response_text = str(state.get("final_reply", ""))
        escalation_reason = str(state.get("triage_reason") or "human_escalation") if escalated else None

    return AdvisorResult(
        request_id=request_id,
        resolved=not escalated and result.get("status") == "done",
        escalated=escalated,
        automated=not escalated,
        response_text=response_text,
        escalation_reason=escalation_reason,
        handling_duration=handling_duration,
    )


class HelpPilotAdvisor:
    """Thin boundary adapter; calling the default runner requires HelpPilot's API key."""

    def __init__(self, runner: HelpPilotRunner | None = None) -> None:
        self._runner = runner or self._default_runner

    @staticmethod
    def _default_runner(message: str, customer_id: str, request_id: str) -> dict[str, Any]:
        from helppilot.graph import run_turn

        return run_turn(message, customer_id, request_id)

    def handle_support_request(
        self,
        request: SupportRequest,
        customer: CustomerProfile | None = None,
    ) -> AdvisorResult:
        if customer is not None and customer.customer_id != request.customer_id:
            raise ValueError("customer profile ID must match the support request customer ID")

        started = perf_counter()
        result = self._runner(request.customer_message, request.customer_id, request.request_id)
        elapsed = perf_counter() - started
        return map_helppilot_result(request.request_id, result, elapsed)
