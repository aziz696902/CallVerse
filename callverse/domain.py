"""Validated, implementation-neutral contracts shared across CallVerse."""

from __future__ import annotations

from enum import Enum
from math import isclose

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StringEnum(str, Enum):
    """String-backed enum with stable serialization and readable display."""

    def __str__(self) -> str:
        return self.value


class CustomerPersona(StringEnum):
    NEW = "new"
    LOYAL = "loyal"
    UNHAPPY = "unhappy"
    PREMIUM = "premium"
    AT_RISK = "at_risk"


class RequestIntent(StringEnum):
    TRACKING = "tracking"
    REFUND = "refund"
    DAMAGED_ITEM = "damaged_item"
    ADDRESS_CHANGE = "address_change"
    CANCEL_ORDER = "cancel_order"
    PAYMENT_ISSUE = "payment_issue"
    COMPLAINT = "complaint"
    GENERAL = "general"


class Urgency(StringEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Channel(StringEnum):
    TEXT = "text"
    VOICE = "voice"


class KnowledgeBaseState(StringEnum):
    HEALTHY = "healthy"
    PARTIAL = "partial"
    OUTDATED = "outdated"


class CustomerTier(StringEnum):
    STANDARD = "standard"
    PREMIUM = "premium"


class DomainModel(BaseModel):
    """Strict base model so misspelled simulation inputs fail immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportRequest(DomainModel):
    request_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    order_id: str | None = Field(default=None, min_length=1)
    intent: RequestIntent
    urgency: Urgency = Urgency.NORMAL
    channel: Channel = Channel.TEXT
    customer_message: str = Field(min_length=1)
    arrival_time: float = Field(
        ge=0,
        description="Simulated minutes elapsed since the scenario started.",
    )


class AdvisorResult(DomainModel):
    request_id: str = Field(min_length=1)
    resolved: bool
    escalated: bool
    automated: bool
    response_text: str
    escalation_reason: str | None = None
    handling_duration: float | None = Field(
        default=None,
        ge=0,
        description="Handling duration in simulated or wall-clock seconds.",
    )


class CustomerProfile(DomainModel):
    customer_id: str = Field(min_length=1)
    persona: CustomerPersona
    tier: CustomerTier = CustomerTier.STANDARD
    patience_seconds: float | None = Field(default=None, gt=0)


class KpiSnapshot(DomainModel):
    """Center-level metrics; all values remain absent until computed."""

    queue_size: int | None = Field(default=None, ge=0)
    average_waiting_time: float | None = Field(default=None, ge=0)
    sla: float | None = Field(default=None, ge=0, le=1)
    abandonment_rate: float | None = Field(default=None, ge=0, le=1)
    average_handling_time: float | None = Field(default=None, ge=0)
    first_contact_resolution: float | None = Field(default=None, ge=0, le=1)
    occupancy: float | None = Field(default=None, ge=0, le=1)
    customer_satisfaction: float | None = Field(default=None, ge=0, le=1)
    operating_cost: float | None = Field(default=None, ge=0)


class ScenarioConfig(DomainModel):
    name: str = Field(min_length=1)
    description: str | None = None
    simulation_duration: float = Field(gt=0, description="Duration in simulated minutes.")
    simulation_start_minute_of_day: int = Field(
        default=480,
        ge=0,
        lt=1440,
        description="Local clock minute corresponding to simulation time zero.",
    )
    random_seed: int = Field(default=42, ge=0)
    external_condition: str = Field(default="normal", min_length=1)
    demand_multiplier: float = Field(default=1.0, gt=0)
    late_delivery_rate: float = Field(default=0.0, ge=0, le=1)
    available_agents: int = Field(gt=0)
    customer_persona_mix: dict[CustomerPersona, float]
    request_intent_mix: dict[RequestIntent, float]
    knowledge_base_state: KnowledgeBaseState = KnowledgeBaseState.HEALTHY
    ai_advisor_enabled: bool = True

    @field_validator("customer_persona_mix", "request_intent_mix")
    @classmethod
    def validate_probability_mix(cls, mix: dict[Enum, float]) -> dict[Enum, float]:
        if not mix:
            raise ValueError("probability mix must not be empty")
        if any(probability < 0 or probability > 1 for probability in mix.values()):
            raise ValueError("each mix probability must be between 0 and 1")
        total = sum(mix.values())
        if not isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"mix probabilities must sum to 1.0; got {total:.6f}")
        return mix
