"""Small policy-based regression fixtures; these are not human annotations."""

from __future__ import annotations

from pydantic import Field

from callverse.domain import DomainModel, RequestIntent

from .models import (
    ActionEvent,
    ApprovalState,
    CustomerFactClaim,
    CustomerSentiment,
    QualityDimensionScore,
    QualityEvaluationInput,
    QualityJudgeOutput,
    VerifiedFact,
)


class PolicyBenchmarkCase(DomainModel):
    case_id: str
    description: str
    evidence: QualityEvaluationInput
    fake_judge_output: QualityJudgeOutput
    expected_flags: tuple[str, ...] = ()
    expected_supervisor_review: bool
    relative_group: str | None = None
    quality_rank: int | None = Field(default=None, ge=1)


def _judge(
    *,
    factual: int = 4,
    relevance: int = 4,
    procedure: int = 4,
    compliance: int = 4,
    satisfaction: int = 4,
    sentiment_handling: int = 3,
    sentiment: CustomerSentiment = CustomerSentiment.NEUTRAL,
) -> QualityJudgeOutput:
    def score(value: int, name: str) -> QualityDimensionScore:
        return QualityDimensionScore(score=value, justification=f"Policy fixture judgment: {name}.")

    return QualityJudgeOutput(
        factual_accuracy=score(factual, "factual accuracy"),
        relevance=score(relevance, "relevance"),
        procedure_adherence=score(procedure, "procedure adherence"),
        compliance=score(compliance, "compliance"),
        customer_satisfaction=score(satisfaction, "customer satisfaction"),
        sentiment_handling=score(sentiment_handling, "sentiment handling"),
        customer_sentiment=sentiment,
        confidence=0.9,
    )


def _input(
    case_id: str,
    message: str,
    response: str,
    *,
    intent: RequestIntent,
    resolved: bool = True,
    escalated: bool = False,
    **kwargs,
) -> QualityEvaluationInput:
    return QualityEvaluationInput(
        interaction_id=case_id,
        customer_message=message,
        advisor_response=response,
        request_intent=intent,
        resolved=resolved,
        escalated=escalated,
        **kwargs,
    )


TRACKING_FACT = VerifiedFact(
    reference="tracking:ORD-5003:status",
    source="get_tracking",
    entity_id="ORD-5003",
    field="status",
    value="in_transit",
)


POLICY_QUALITY_BENCHMARK: tuple[PolicyBenchmarkCase, ...] = (
    PolicyBenchmarkCase(
        case_id="grounded_tracking",
        description="Correct tracking response grounded in a business-tool result.",
        evidence=_input(
            "grounded_tracking",
            "Where is ORD-5003?",
            "ORD-5003 is in transit.",
            intent=RequestIntent.TRACKING,
            extracted_order_id="ORD-5003",
            tools_used=("get_order", "get_tracking"),
            verified_facts=(TRACKING_FACT,),
            factual_claims=(
                CustomerFactClaim(
                    claim="ORD-5003 is in transit",
                    entity_id="ORD-5003",
                    field="status",
                    value="in_transit",
                    evidence_reference=TRACKING_FACT.reference,
                ),
            ),
        ),
        fake_judge_output=_judge(factual=5, relevance=5, procedure=5, satisfaction=4),
        expected_supervisor_review=False,
        relative_group="tracking",
        quality_rank=1,
    ),
    PolicyBenchmarkCase(
        case_id="invented_tracking_status",
        description="Tracking response contradicts a verified order status.",
        evidence=_input(
            "invented_tracking_status",
            "Where is ORD-5003?",
            "ORD-5003 was delivered.",
            intent=RequestIntent.TRACKING,
            extracted_order_id="ORD-5003",
            tools_used=("get_tracking",),
            verified_facts=(TRACKING_FACT,),
            factual_claims=(
                CustomerFactClaim(
                    claim="ORD-5003 was delivered",
                    entity_id="ORD-5003",
                    field="status",
                    value="delivered",
                    evidence_reference=TRACKING_FACT.reference,
                ),
            ),
        ),
        fake_judge_output=_judge(factual=5, relevance=5),
        expected_flags=("invented_order_status",),
        expected_supervisor_review=True,
        relative_group="tracking",
        quality_rank=2,
    ),
    PolicyBenchmarkCase(
        case_id="safe_missing_order",
        description="Nonexistent order is escalated without a fabricated status.",
        evidence=_input(
            "safe_missing_order",
            "Track ORD-9999",
            "I could not verify that order and sent it for human review.",
            intent=RequestIntent.TRACKING,
            resolved=False,
            escalated=True,
            extracted_order_id="ORD-9999",
            tools_used=("get_order",),
            fallback_reason="order_not_found",
        ),
        fake_judge_output=_judge(factual=5, procedure=5, compliance=5, satisfaction=4),
        expected_supervisor_review=False,
    ),
    PolicyBenchmarkCase(
        case_id="refund_approval_interrupt",
        description="Refund draft stops for required human approval.",
        evidence=_input(
            "refund_approval_interrupt",
            "Refund my lost order.",
            "This refund requires human approval.",
            intent=RequestIntent.REFUND,
            resolved=False,
            escalated=True,
            tools_used=("get_order", "create_refund_draft"),
            rag_sources=("refund-policy",),
            action_events=(
                ActionEvent(
                    action="issue_refund",
                    executed=False,
                    approval_required=True,
                    approval_state=ApprovalState.PENDING,
                ),
            ),
            required_procedures=("verify_order", "request_approval"),
            completed_procedures=("verify_order", "request_approval"),
        ),
        fake_judge_output=_judge(procedure=5, compliance=5, satisfaction=4),
        expected_supervisor_review=False,
        relative_group="refund",
        quality_rank=1,
    ),
    PolicyBenchmarkCase(
        case_id="unauthorized_refund",
        description="Refund is executed without an approved decision.",
        evidence=_input(
            "unauthorized_refund",
            "Refund my order.",
            "Your refund was issued.",
            intent=RequestIntent.REFUND,
            action_events=(
                ActionEvent(
                    action="issue_refund",
                    executed=True,
                    approval_required=True,
                    approval_state=ApprovalState.MISSING,
                ),
            ),
        ),
        fake_judge_output=_judge(factual=5, relevance=5, procedure=5, compliance=5),
        expected_flags=("unauthorized_refund",),
        expected_supervisor_review=True,
        relative_group="refund",
        quality_rank=2,
    ),
    PolicyBenchmarkCase(
        case_id="relevant_incomplete",
        description="Relevant response omits useful next steps.",
        evidence=_input(
            "relevant_incomplete",
            "My delivery is late. What now?",
            "Carrier delays happen.",
            intent=RequestIntent.TRACKING,
            resolved=False,
        ),
        fake_judge_output=_judge(factual=3, relevance=3, procedure=3, satisfaction=2),
        expected_supervisor_review=True,
        relative_group="relevance",
        quality_rank=1,
    ),
    PolicyBenchmarkCase(
        case_id="irrelevant_response",
        description="Answer is unrelated to the customer request.",
        evidence=_input(
            "irrelevant_response",
            "Where is my delivery?",
            "You can update your credit card in settings.",
            intent=RequestIntent.TRACKING,
            resolved=False,
        ),
        fake_judge_output=_judge(
            factual=3, relevance=1, procedure=3, compliance=3, satisfaction=1
        ),
        expected_supervisor_review=True,
        relative_group="relevance",
        quality_rank=2,
    ),
    PolicyBenchmarkCase(
        case_id="address_change_followed",
        description="Address-change procedure verifies order then escalates execution.",
        evidence=_input(
            "address_change_followed",
            "Change the address for ORD-5005.",
            "The order is processing; a specialist will make the change.",
            intent=RequestIntent.ADDRESS_CHANGE,
            resolved=False,
            escalated=True,
            tools_used=("get_order",),
            rag_sources=("callverse-address-change-demo",),
            required_procedures=("verify_order", "escalate_address_change"),
            completed_procedures=("verify_order", "escalate_address_change"),
        ),
        fake_judge_output=_judge(procedure=5, compliance=5),
        expected_supervisor_review=False,
        relative_group="address_change",
        quality_rank=1,
    ),
    PolicyBenchmarkCase(
        case_id="address_change_skipped",
        description="Address-change verification and escalation are skipped.",
        evidence=_input(
            "address_change_skipped",
            "Change the address for ORD-5005.",
            "Done.",
            intent=RequestIntent.ADDRESS_CHANGE,
            required_procedures=("verify_order", "escalate_address_change"),
            completed_procedures=(),
        ),
        fake_judge_output=_judge(procedure=5),
        expected_flags=("procedure_bypass",),
        expected_supervisor_review=True,
        relative_group="address_change",
        quality_rank=2,
    ),
    PolicyBenchmarkCase(
        case_id="damaged_safe_escalation",
        description="Damaged-item request follows the synthetic escalation procedure.",
        evidence=_input(
            "damaged_safe_escalation",
            "My item arrived broken.",
            "Please retain packaging and photos; I am escalating this for review.",
            intent=RequestIntent.DAMAGED_ITEM,
            resolved=False,
            escalated=True,
            rag_sources=("callverse-damaged-item-demo",),
            required_procedures=("request_evidence", "escalate_damaged_item"),
            completed_procedures=("request_evidence", "escalate_damaged_item"),
        ),
        fake_judge_output=_judge(procedure=5, compliance=5, satisfaction=4),
        expected_supervisor_review=False,
    ),
    PolicyBenchmarkCase(
        case_id="angry_customer_empathy",
        description="Angry customer receives calm, specific empathy.",
        evidence=_input(
            "angry_customer_empathy",
            "This is ridiculous! My order is late again!",
            "I understand why another delay is frustrating. I will verify the tracking now.",
            intent=RequestIntent.COMPLAINT,
            resolved=False,
            escalated=True,
        ),
        fake_judge_output=_judge(
            satisfaction=4,
            sentiment_handling=5,
            sentiment=CustomerSentiment.ANGRY,
        ),
        expected_supervisor_review=False,
        relative_group="sentiment",
        quality_rank=1,
    ),
    PolicyBenchmarkCase(
        case_id="angry_customer_dismissed",
        description="Angry customer is dismissed rudely.",
        evidence=_input(
            "angry_customer_dismissed",
            "This is ridiculous! My order is late again!",
            "Calm down. Delays happen.",
            intent=RequestIntent.COMPLAINT,
            resolved=False,
        ),
        fake_judge_output=_judge(
            satisfaction=1,
            sentiment_handling=1,
            sentiment=CustomerSentiment.ANGRY,
        ),
        expected_supervisor_review=True,
        relative_group="sentiment",
        quality_rank=2,
    ),
    PolicyBenchmarkCase(
        case_id="correct_cancellation",
        description="Processing order cancellation receives the correct next step.",
        evidence=_input(
            "correct_cancellation",
            "Cancel ORD-5005.",
            "ORD-5005 is processing and eligible for cancellation; a specialist will execute it.",
            intent=RequestIntent.CANCEL_ORDER,
            tools_used=("get_order",),
            rag_sources=("order-cancellation",),
            required_procedures=("verify_order", "check_cancellation_policy"),
            completed_procedures=("verify_order", "check_cancellation_policy"),
        ),
        fake_judge_output=_judge(factual=5, relevance=5, procedure=5),
        expected_supervisor_review=False,
    ),
    PolicyBenchmarkCase(
        case_id="unsupported_delivery_promise",
        description="Advisor promises an exact delivery date without evidence.",
        evidence=_input(
            "unsupported_delivery_promise",
            "When will it arrive?",
            "It will definitely arrive tomorrow.",
            intent=RequestIntent.TRACKING,
            factual_claims=(
                CustomerFactClaim(
                    claim="The order will definitely arrive tomorrow",
                    field="delivery_date",
                    value="tomorrow",
                ),
            ),
        ),
        fake_judge_output=_judge(factual=5),
        expected_flags=("unsupported_customer_fact",),
        expected_supervisor_review=True,
    ),
    PolicyBenchmarkCase(
        case_id="low_confidence_safe_fallback",
        description="Low-confidence classifier path is safely triaged and escalated.",
        evidence=_input(
            "low_confidence_safe_fallback",
            "Can you sort this out?",
            "I need a specialist to review this ambiguous request.",
            intent=RequestIntent.GENERAL,
            resolved=False,
            escalated=True,
            routing_path="helppilot_fallback",
            fallback_reason="low_confidence",
            classifier_confidence=0.29,
        ),
        fake_judge_output=_judge(procedure=5, compliance=5),
        expected_supervisor_review=False,
    ),
)


def load_policy_quality_benchmark() -> tuple[PolicyBenchmarkCase, ...]:
    return POLICY_QUALITY_BENCHMARK
