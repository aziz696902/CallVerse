"""Typed, observable contracts for CallVerse quality evaluation."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from callverse.domain import DomainModel, RequestIntent

if TYPE_CHECKING:
    from callverse.customer_advisor import AdvisorInteraction


class QualityDimension(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    RELEVANCE = "relevance"
    PROCEDURE_ADHERENCE = "procedure_adherence"
    COMPLIANCE = "compliance"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    SENTIMENT_HANDLING = "sentiment_handling"


class QualitySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"


class CustomerSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MISSING = "missing"


class VerifiedFact(DomainModel):
    reference: str = Field(min_length=1)
    source: str = Field(min_length=1)
    entity_id: str | None = None
    field: str = Field(min_length=1)
    value: str = Field(min_length=1)


class CustomerFactClaim(DomainModel):
    claim: str = Field(min_length=1)
    entity_id: str | None = None
    field: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence_reference: str | None = None


class ActionEvent(DomainModel):
    action: str = Field(min_length=1)
    executed: bool
    approval_required: bool = False
    approval_state: ApprovalState = ApprovalState.NOT_REQUIRED

    @model_validator(mode="after")
    def validate_approval_state(self) -> ActionEvent:
        if self.approval_required and self.approval_state is ApprovalState.NOT_REQUIRED:
            raise ValueError("approval-required actions need an observable approval state")
        return self


class QualityEvaluationInput(DomainModel):
    interaction_id: str = Field(min_length=1)
    customer_message: str = Field(min_length=1)
    advisor_response: str = Field(min_length=1)
    request_intent: RequestIntent | None = None
    classifier_confidence: float | None = Field(default=None, ge=0, le=1)
    extracted_order_id: str | None = None
    resolved: bool
    escalated: bool
    routing_path: str | None = None
    fallback_reason: str | None = None
    tools_used: tuple[str, ...] = ()
    verified_facts: tuple[VerifiedFact, ...] = ()
    factual_claims: tuple[CustomerFactClaim, ...] = ()
    rag_sources: tuple[str, ...] = ()
    action_events: tuple[ActionEvent, ...] = ()
    required_procedures: tuple[str, ...] = ()
    completed_procedures: tuple[str, ...] = ()

    @classmethod
    def from_advisor_interaction(
        cls,
        customer_message: str,
        interaction: AdvisorInteraction,
        *,
        verified_facts: tuple[VerifiedFact, ...] = (),
        factual_claims: tuple[CustomerFactClaim, ...] = (),
        action_events: tuple[ActionEvent, ...] = (),
        required_procedures: tuple[str, ...] = (),
        completed_procedures: tuple[str, ...] = (),
    ) -> QualityEvaluationInput:
        metadata = interaction.metadata
        result = interaction.result
        return cls(
            interaction_id=result.request_id,
            customer_message=customer_message,
            advisor_response=result.response_text,
            request_intent=metadata.predicted_intent,
            classifier_confidence=metadata.classifier_confidence,
            extracted_order_id=metadata.extracted_order_id,
            resolved=result.resolved,
            escalated=result.escalated,
            routing_path=metadata.routing_path.value,
            fallback_reason=metadata.fallback_reason,
            tools_used=metadata.tools_used,
            verified_facts=verified_facts,
            factual_claims=factual_claims,
            rag_sources=metadata.rag_sources,
            action_events=action_events,
            required_procedures=required_procedures,
            completed_procedures=completed_procedures,
        )


class QualityDimensionScore(DomainModel):
    score: int = Field(ge=1, le=5)
    justification: str = Field(min_length=1, max_length=600)


class QualityFlag(DomainModel):
    code: str = Field(min_length=1)
    severity: QualitySeverity
    explanation: str = Field(min_length=1, max_length=600)
    evidence_reference: str | None = None
    deterministic: bool = False


class QualityJudgeOutput(DomainModel):
    factual_accuracy: QualityDimensionScore
    relevance: QualityDimensionScore
    procedure_adherence: QualityDimensionScore
    compliance: QualityDimensionScore
    customer_satisfaction: QualityDimensionScore
    sentiment_handling: QualityDimensionScore
    flags: tuple[QualityFlag, ...] = ()
    customer_sentiment: CustomerSentiment | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class QualityEvaluationResult(DomainModel):
    interaction_id: str
    status: QualityStatus
    factual_accuracy: QualityDimensionScore | None = None
    relevance: QualityDimensionScore | None = None
    procedure_adherence: QualityDimensionScore | None = None
    compliance: QualityDimensionScore | None = None
    customer_satisfaction: QualityDimensionScore | None = None
    sentiment_handling: QualityDimensionScore | None = None
    overall_score: float | None = Field(default=None, ge=1, le=5)
    flags: tuple[QualityFlag, ...] = ()
    requires_supervisor_review: bool
    evaluator_type: Literal["none", "injected_structured", "groq_structured"]
    evaluator_provider: str | None = None
    rubric_version: str
    customer_sentiment: CustomerSentiment | None = None
    evaluator_confidence: float | None = Field(default=None, ge=0, le=1)
    guardrail_observations: tuple[str, ...] = ()
    status_reason: str | None = None
    resolved: bool
    escalated: bool


class QualityWeights(DomainModel):
    factual_accuracy: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    procedure_adherence: float = Field(ge=0, le=1)
    compliance: float = Field(ge=0, le=1)
    customer_satisfaction: float = Field(ge=0, le=1)
    sentiment_handling: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sum_to_one(self) -> QualityWeights:
        if abs(sum(self.as_dict().values()) - 1.0) > 1e-9:
            raise ValueError("quality weights must sum to 1.0")
        return self

    def as_dict(self) -> dict[QualityDimension, float]:
        return {
            QualityDimension.FACTUAL_ACCURACY: self.factual_accuracy,
            QualityDimension.RELEVANCE: self.relevance,
            QualityDimension.PROCEDURE_ADHERENCE: self.procedure_adherence,
            QualityDimension.COMPLIANCE: self.compliance,
            QualityDimension.CUSTOMER_SATISFACTION: self.customer_satisfaction,
            QualityDimension.SENTIMENT_HANDLING: self.sentiment_handling,
        }
