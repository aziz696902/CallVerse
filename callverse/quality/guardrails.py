"""Deterministic checks over explicit interaction evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    QualityDimension,
    QualityEvaluationInput,
    QualityFlag,
    QualitySeverity,
)


@dataclass(frozen=True)
class GuardrailAssessment:
    flags: tuple[QualityFlag, ...]
    score_caps: dict[QualityDimension, int]
    observations: tuple[str, ...]


def _cap(caps: dict[QualityDimension, int], dimension: QualityDimension, value: int) -> None:
    caps[dimension] = min(caps.get(dimension, 5), value)


def evaluate_guardrails(evidence: QualityEvaluationInput) -> GuardrailAssessment:
    flags: list[QualityFlag] = []
    caps: dict[QualityDimension, int] = {}
    observations: list[str] = []

    facts_by_reference = {fact.reference: fact for fact in evidence.verified_facts}
    for claim in evidence.factual_claims:
        fact = facts_by_reference.get(claim.evidence_reference or "")
        if fact is None:
            code = "invented_order_status" if claim.field == "status" else "unsupported_customer_fact"
            flags.append(
                QualityFlag(
                    code=code,
                    severity=QualitySeverity.HIGH,
                    explanation=f"Customer-specific claim lacks structured business evidence: {claim.claim}",
                    evidence_reference=claim.evidence_reference,
                    deterministic=True,
                )
            )
            _cap(caps, QualityDimension.FACTUAL_ACCURACY, 1 if claim.field == "status" else 2)
        elif (
            fact.field != claim.field
            or fact.value.casefold() != claim.value.casefold()
            or (claim.entity_id and fact.entity_id and claim.entity_id != fact.entity_id)
        ):
            flags.append(
                QualityFlag(
                    code="invented_order_status" if claim.field == "status" else "contradictory_customer_fact",
                    severity=QualitySeverity.HIGH,
                    explanation=f"Claim contradicts verified evidence: {claim.claim}",
                    evidence_reference=fact.reference,
                    deterministic=True,
                )
            )
            _cap(caps, QualityDimension.FACTUAL_ACCURACY, 1)
        else:
            observations.append(f"grounded_claim:{fact.reference}")

    for action in evidence.action_events:
        account_change = action.action in {"change_shipping_address", "update_payment_method"}
        if action.executed and action.approval_required and action.approval_state.value != "approved":
            if action.action == "issue_refund":
                code = "unauthorized_refund"
            elif account_change:
                code = "unsafe_account_change"
            else:
                code = "missing_required_approval"
            flags.append(
                QualityFlag(
                    code=code,
                    severity=QualitySeverity.CRITICAL,
                    explanation=(
                        f"{action.action} was executed without an observable approved decision."
                    ),
                    evidence_reference=f"action:{action.action}",
                    deterministic=True,
                )
            )
            _cap(caps, QualityDimension.COMPLIANCE, 1)
            _cap(caps, QualityDimension.PROCEDURE_ADHERENCE, 1)
        elif action.approval_required and not action.executed:
            observations.append(f"approval_respected:{action.action}:{action.approval_state.value}")

    missing_procedures = sorted(set(evidence.required_procedures) - set(evidence.completed_procedures))
    if missing_procedures:
        flags.append(
            QualityFlag(
                code="procedure_bypass",
                severity=QualitySeverity.MEDIUM,
                explanation="Required procedure events were not observed: " + ", ".join(missing_procedures),
                evidence_reference="required_procedures",
                deterministic=True,
            )
        )
        _cap(caps, QualityDimension.PROCEDURE_ADHERENCE, 2)

    if evidence.fallback_reason in {"order_not_found", "order_customer_mismatch", "order_lookup_failure"}:
        if evidence.escalated and not evidence.factual_claims:
            observations.append("safe_missing_order_escalation")
        elif not evidence.escalated:
            flags.append(
                QualityFlag(
                    code="unsafe_missing_order_handling",
                    severity=QualitySeverity.HIGH,
                    explanation="Missing or unverifiable order data was not escalated.",
                    evidence_reference="fallback_reason",
                    deterministic=True,
                )
            )
            _cap(caps, QualityDimension.PROCEDURE_ADHERENCE, 1)

    return GuardrailAssessment(tuple(flags), caps, tuple(observations))
