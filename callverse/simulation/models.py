"""Compact result contracts produced by the simulation engine."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from callverse.domain import CustomerPersona, DomainModel, KpiSnapshot, RequestIntent


class RequestOutcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REMAINING = "remaining"


class RequestCounts(DomainModel):
    generated: int = Field(ge=0)
    completed: int = Field(ge=0)
    abandoned: int = Field(ge=0)
    remaining: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_accounting(self) -> RequestCounts:
        if self.generated != self.completed + self.abandoned + self.remaining:
            raise ValueError("generated requests must equal completed + abandoned + remaining")
        return self


class RequestEventRecord(DomainModel):
    request_id: str
    customer_id: str
    persona: CustomerPersona
    intent: RequestIntent
    arrival_time: float = Field(ge=0)
    patience_minutes: float = Field(gt=0)
    handling_minutes: float = Field(gt=0)
    service_start: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    waiting_time: float = Field(ge=0)
    outcome: RequestOutcome


class TimeSeriesSnapshot(DomainModel):
    simulation_time: float = Field(ge=0)
    queue_size: int = Field(ge=0)
    busy_agents: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    abandoned_count: int = Field(ge=0)


class SimulationResult(DomainModel):
    scenario_name: str
    seed: int = Field(ge=0)
    duration: float = Field(gt=0)
    available_agents: int = Field(gt=0)
    ai_advisor_enabled: bool
    sla_target_minutes: float = Field(ge=0)
    busy_advisor_minutes: float = Field(ge=0)
    capacity_advisor_minutes: float = Field(gt=0)
    counts: RequestCounts
    kpis: KpiSnapshot
    intent_counts: dict[RequestIntent, int]
    persona_counts: dict[CustomerPersona, int]
    snapshots: tuple[TimeSeriesSnapshot, ...]
    request_records: tuple[RequestEventRecord, ...]
    event_records_truncated: bool = False

    @model_validator(mode="after")
    def validate_result_consistency(self) -> SimulationResult:
        if self.busy_advisor_minutes > self.capacity_advisor_minutes + 1e-9:
            raise ValueError("busy advisor time cannot exceed total advisor capacity")
        if sum(self.intent_counts.values()) != self.counts.generated:
            raise ValueError("intent counts must sum to generated requests")
        if sum(self.persona_counts.values()) != self.counts.generated:
            raise ValueError("persona counts must sum to generated requests")
        if any(snapshot.busy_agents > self.available_agents for snapshot in self.snapshots):
            raise ValueError("snapshot busy-agent count cannot exceed available agents")
        return self


class ScenarioComparison(DomainModel):
    scenario_name: str
    generated_requests: int = Field(ge=0)
    average_waiting_time: float | None = Field(default=None, ge=0)
    sla: float | None = Field(default=None, ge=0, le=1)
    abandonment_rate: float = Field(ge=0, le=1)
    occupancy: float = Field(ge=0, le=1)
