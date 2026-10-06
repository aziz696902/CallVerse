from __future__ import annotations

from itertools import pairwise

import pytest
from pydantic import ValidationError

from callverse.customer_advisor import (
    AdvisorInteraction,
    InteractionMetadata,
    RoutingPath,
)
from callverse.domain import AdvisorResult, RequestIntent
from callverse.quality import (
    DEFAULT_WEIGHTS,
    ActionEvent,
    ApprovalState,
    CustomerFactClaim,
    QualityAnalyst,
    QualityDimension,
    QualityDimensionScore,
    QualityEvaluationInput,
    QualityFlag,
    QualityJudgeOutput,
    QualitySeverity,
    QualityStatus,
    QualityWeights,
    VerifiedFact,
    aggregate_quality,
    calculate_overall_score,
    evaluate_guardrails,
)
from callverse.quality.benchmark import load_policy_quality_benchmark


def basic_input(**updates) -> QualityEvaluationInput:
    values = {
        "interaction_id": "QUALITY-TEST",
        "customer_message": "Where is ORD-5003?",
        "advisor_response": "ORD-5003 is in transit.",
        "request_intent": RequestIntent.TRACKING,
        "resolved": True,
        "escalated": False,
    }
    values.update(updates)
    return QualityEvaluationInput(**values)


def judge_output(score: int = 4) -> QualityJudgeOutput:
    dimension = QualityDimensionScore(score=score, justification="Observable fixture evidence supports this score.")
    return QualityJudgeOutput(
        factual_accuracy=dimension,
        relevance=dimension,
        procedure_adherence=dimension,
        compliance=dimension,
        customer_satisfaction=dimension,
        sentiment_handling=dimension,
        confidence=0.8,
    )


def evaluate_case(case):
    return QualityAnalyst(judge=lambda *_: case.fake_judge_output).evaluate(case.evidence)


def test_weights_sum_to_one_and_match_v1_policy():
    weights = DEFAULT_WEIGHTS.as_dict()

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[QualityDimension.FACTUAL_ACCURACY] == 0.25
    assert weights[QualityDimension.PROCEDURE_ADHERENCE] == 0.20
    assert weights[QualityDimension.COMPLIANCE] == 0.20


def test_invalid_weights_are_rejected():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        QualityWeights(
            factual_accuracy=0.5,
            relevance=0.5,
            procedure_adherence=0.5,
            compliance=0.5,
            customer_satisfaction=0.5,
            sentiment_handling=0.5,
        )


def test_dimension_score_bounds_are_enforced():
    with pytest.raises(ValidationError):
        QualityDimensionScore(score=0, justification="Too low")
    with pytest.raises(ValidationError):
        QualityDimensionScore(score=6, justification="Too high")


def test_compliance_flag_validation_is_strict():
    with pytest.raises(ValidationError):
        QualityFlag(code="", severity=QualitySeverity.HIGH, explanation="Invalid")
    with pytest.raises(ValidationError):
        QualityFlag(code="test", severity="cosmetic", explanation="Invalid")


def test_deterministic_overall_uses_configured_weights():
    scores = {
        QualityDimension.FACTUAL_ACCURACY: QualityDimensionScore(score=5, justification="x"),
        QualityDimension.PROCEDURE_ADHERENCE: QualityDimensionScore(score=4, justification="x"),
        QualityDimension.COMPLIANCE: QualityDimensionScore(score=3, justification="x"),
        QualityDimension.RELEVANCE: QualityDimensionScore(score=2, justification="x"),
        QualityDimension.CUSTOMER_SATISFACTION: QualityDimensionScore(score=1, justification="x"),
        QualityDimension.SENTIMENT_HANDLING: QualityDimensionScore(score=5, justification="x"),
    }

    assert calculate_overall_score(scores) == 3.55


def test_missing_dimension_cannot_produce_overall_score():
    with pytest.raises(ValueError, match="missing quality dimensions"):
        calculate_overall_score({})


def test_grounded_tracking_fact_has_no_guardrail_flag():
    fact = VerifiedFact(
        reference="tracking:1", source="get_tracking", entity_id="ORD-1", field="status", value="in_transit"
    )
    evidence = basic_input(
        verified_facts=(fact,),
        factual_claims=(
            CustomerFactClaim(
                claim="Order is in transit",
                entity_id="ORD-1",
                field="status",
                value="in_transit",
                evidence_reference="tracking:1",
            ),
        ),
    )

    assessment = evaluate_guardrails(evidence)

    assert assessment.flags == ()
    assert assessment.observations == ("grounded_claim:tracking:1",)


def test_hallucinated_order_status_gets_high_severity_flag():
    case = next(case for case in load_policy_quality_benchmark() if case.case_id == "invented_tracking_status")

    assessment = evaluate_guardrails(case.evidence)

    assert assessment.flags[0].code == "invented_order_status"
    assert assessment.flags[0].severity is QualitySeverity.HIGH
    assert assessment.score_caps[QualityDimension.FACTUAL_ACCURACY] == 1


def test_safe_missing_order_escalation_is_not_penalized():
    case = next(case for case in load_policy_quality_benchmark() if case.case_id == "safe_missing_order")

    result = evaluate_case(case)

    assert result.flags == ()
    assert "safe_missing_order_escalation" in result.guardrail_observations
    assert result.requires_supervisor_review is False


def test_refund_approval_interruption_is_valid_procedure():
    case = next(
        case for case in load_policy_quality_benchmark() if case.case_id == "refund_approval_interrupt"
    )

    result = evaluate_case(case)

    assert result.flags == ()
    assert any(item.startswith("approval_respected:issue_refund") for item in result.guardrail_observations)
    assert result.compliance.score == 5


def test_unauthorized_refund_is_critical_and_caps_favorable_judge():
    case = next(case for case in load_policy_quality_benchmark() if case.case_id == "unauthorized_refund")

    result = evaluate_case(case)

    assert "unauthorized_refund" in {flag.code for flag in result.flags}
    assert result.compliance.score == 1
    assert result.procedure_adherence.score == 1
    assert result.requires_supervisor_review is True


def test_unapproved_account_change_uses_specific_compliance_flag():
    evidence = basic_input(
        action_events=(
            ActionEvent(
                action="change_shipping_address",
                executed=True,
                approval_required=True,
                approval_state=ApprovalState.MISSING,
            ),
        )
    )

    assessment = evaluate_guardrails(evidence)

    assert assessment.flags[0].code == "unsafe_account_change"
    assert assessment.flags[0].severity is QualitySeverity.CRITICAL


def test_irrelevant_response_loses_relevance():
    case = next(case for case in load_policy_quality_benchmark() if case.case_id == "irrelevant_response")

    result = evaluate_case(case)

    assert result.relevance.score == 1
    assert result.requires_supervisor_review is True


def test_procedure_violation_is_deterministically_capped():
    case = next(case for case in load_policy_quality_benchmark() if case.case_id == "address_change_skipped")

    result = evaluate_case(case)

    assert "procedure_bypass" in {flag.code for flag in result.flags}
    assert result.procedure_adherence.score == 2


def test_angry_customer_handling_has_expected_relative_quality():
    cases = {case.case_id: case for case in load_policy_quality_benchmark()}
    empathetic = evaluate_case(cases["angry_customer_empathy"])
    dismissed = evaluate_case(cases["angry_customer_dismissed"])

    assert empathetic.sentiment_handling.score == 5
    assert dismissed.sentiment_handling.score == 1
    assert empathetic.overall_score > dismissed.overall_score


def test_judge_unavailable_preserves_guardrails_without_fabricating_scores():
    evidence = basic_input(
        action_events=(
            ActionEvent(
                action="issue_refund",
                executed=True,
                approval_required=True,
                approval_state=ApprovalState.MISSING,
            ),
        )
    )

    result = QualityAnalyst().evaluate(evidence)

    assert result.status is QualityStatus.UNAVAILABLE
    assert result.overall_score is None
    assert result.factual_accuracy is None
    assert "unauthorized_refund" in {flag.code for flag in result.flags}
    assert result.requires_supervisor_review is True


def test_malformed_judge_output_fails_safely():
    result = QualityAnalyst(judge=lambda *_: {"relevance": {"score": 9}}).evaluate(basic_input())

    assert result.status is QualityStatus.UNAVAILABLE
    assert result.overall_score is None
    assert result.requires_supervisor_review is True
    assert "ValidationError" in result.status_reason


def test_fake_judge_receives_factual_evidence_and_rubric():
    captured = {}

    def judge(evidence, rubric):
        captured["evidence"] = evidence
        captured["rubric"] = rubric
        return judge_output()

    fact = VerifiedFact(reference="order:1", source="get_order", field="status", value="processing")
    result = QualityAnalyst(judge=judge).evaluate(basic_input(verified_facts=(fact,)))

    assert captured["evidence"].verified_facts == (fact,)
    assert "factual_accuracy" in captured["rubric"]
    assert result.status is QualityStatus.COMPLETED
    assert result.overall_score == 4.0


def test_phase6_interaction_maps_to_quality_evidence_without_hidden_trace():
    interaction = AdvisorInteraction(
        result=AdvisorResult(
            request_id="REQ-P6",
            resolved=True,
            escalated=False,
            automated=True,
            response_text="ORD-5003 is in transit.",
        ),
        metadata=InteractionMetadata(
            routing_path=RoutingPath.CLASSIFIER,
            predicted_intent=RequestIntent.TRACKING,
            classifier_confidence=0.91,
            classifier_accepted=True,
            extracted_order_id="ORD-5003",
            tools_used=("get_order", "get_tracking"),
            rag_sources=("shipping-delivery",),
            resolved=True,
            escalated=False,
        ),
    )

    evidence = QualityEvaluationInput.from_advisor_interaction("Track ORD-5003", interaction)

    assert evidence.interaction_id == "REQ-P6"
    assert evidence.request_intent is RequestIntent.TRACKING
    assert evidence.tools_used == ("get_order", "get_tracking")
    assert evidence.rag_sources == ("shipping-delivery",)


def test_hard_flag_cannot_be_erased_by_favorable_judge_flags():
    evidence = basic_input(
        factual_claims=(
            CustomerFactClaim(claim="It arrives tomorrow", field="delivery_date", value="tomorrow"),
        )
    )
    favorable = judge_output(score=5)

    result = QualityAnalyst(judge=lambda *_: favorable).evaluate(evidence)

    assert "unsupported_customer_fact" in {flag.code for flag in result.flags}
    assert result.factual_accuracy.score == 2
    assert result.requires_supervisor_review is True


def test_policy_benchmark_has_compact_required_coverage():
    cases = load_policy_quality_benchmark()
    identifiers = {case.case_id for case in cases}

    assert len(cases) == 15
    assert len(identifiers) == len(cases)
    assert {
        "grounded_tracking",
        "invented_tracking_status",
        "safe_missing_order",
        "refund_approval_interrupt",
        "unauthorized_refund",
        "angry_customer_empathy",
        "angry_customer_dismissed",
        "low_confidence_safe_fallback",
    } <= identifiers


def test_all_policy_benchmark_expectations_and_relative_orderings_hold():
    cases = load_policy_quality_benchmark()
    results = {case.case_id: evaluate_case(case) for case in cases}
    for case in cases:
        result = results[case.case_id]
        assert set(case.expected_flags) <= {flag.code for flag in result.flags}
        assert result.requires_supervisor_review is case.expected_supervisor_review

    for group in {case.relative_group for case in cases if case.relative_group}:
        ranked = sorted(
            (case for case in cases if case.relative_group == group), key=lambda case: case.quality_rank
        )
        assert all(
            results[better.case_id].overall_score > results[worse.case_id].overall_score
            for better, worse in pairwise(ranked)
        )


def test_aggregate_quality_produces_dashboard_ready_summary():
    cases = load_policy_quality_benchmark()[:3]
    results = [evaluate_case(case) for case in cases]
    results.append(QualityAnalyst().evaluate(basic_input(interaction_id="unavailable")))

    summary = aggregate_quality(results)

    assert summary["interaction_count"] == 4
    assert summary["evaluated_count"] == 3
    assert summary["unavailable_count"] == 1
    assert summary["average_overall_quality"] is not None
    assert set(summary["average_by_dimension"]) == {item.value for item in QualityDimension}
    assert summary["requires_supervisor_review_count"] >= 2
    assert summary["compliance_flags_by_severity"]["high"] == 1
