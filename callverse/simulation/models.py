"""Compact result contracts produced by the simulation engine."""

from __future__ import annotations

from enum import Enum
from itertools import pairwise

from pydantic import Field, model_validator

from callverse.domain import CustomerPersona, DomainModel, KpiSnapshot, RequestIntent


class RequestOutcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REMAINING = "remaining"


class StaffingMode(str, Enum):
    FIXED = "fixed"
    SCHEDULED = "scheduled"


class StaffingSlot(DomainModel):
    start_minute: float = Field(ge=0)
    end_minute: float = Field(gt=0)
    advisors: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_interval(self) -> StaffingSlot:
        if self.end_minute <= self.start_minute:
            raise ValueError("staffing slot end must follow its start")
        return self


class StaffingSchedule(DomainModel):
    slots: tuple[StaffingSlot, ...]
    slot_minutes: float = Field(gt=0)
    horizon_minutes: float = Field(gt=0)
    source: str = Field(min_length=1)
    provenance: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> StaffingSchedule:
        if not self.slots:
            raise ValueError("staffing schedule must contain at least one slot")
        if self.slots[0].start_minute != 0:
            raise ValueError("staffing schedule must begin at simulation minute zero")
        if self.slots[-1].end_minute != self.horizon_minutes:
            raise ValueError("staffing schedule must cover the full simulation horizon")
        if any(
            abs((slot.end_minute - slot.start_minute) - self.slot_minutes) > 1e-9
            for slot in self.slots
        ):
            raise ValueError("every staffing slot must use the declared interval length")
        if any(
            abs(previous.end_minute - current.start_minute) > 1e-9
            for previous, current in pairwise(self.slots)
        ):
            raise ValueError("staffing slots must be continuous and non-overlapping")
        return self

    @property
    def total_agent_hours(self) -> float:
        return sum(
            slot.advisors * (slot.end_minute - slot.start_minute) / 60
            for slot in self.slots
        )

    @property
    def minimum_advisors(self) -> int:
        return min(slot.advisors for slot in self.slots)

    @property
    def maximum_advisors(self) -> int:
        return max(slot.advisors for slot in self.slots)

    @property
    def average_advisors(self) -> float:
        return self.total_agent_hours / (self.horizon_minutes / 60)

    @property
    def staffing_changes(self) -> int:
        return sum(
            left.advisors != right.advisors for left, right in pairwise(self.slots)
        )


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
    available_agents: int = Field(gt=0)
    completed_count: int = Field(ge=0)
    abandoned_count: int = Field(ge=0)

    @property
    def free_agents(self) -> int:
        return max(self.available_agents - self.busy_agents, 0)

    @property
    def overhang_busy_agents(self) -> int:
        return max(self.busy_agents - self.available_agents, 0)


class SimulationResult(DomainModel):
    scenario_name: str
    seed: int = Field(ge=0)
    duration: float = Field(gt=0)
    available_agents: int = Field(gt=0)
    staffing_mode: StaffingMode = StaffingMode.FIXED
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
        if (
            self.staffing_mode is StaffingMode.FIXED
            and self.busy_advisor_minutes > self.capacity_advisor_minutes + 1e-9
        ):
            raise ValueError("busy advisor time cannot exceed total advisor capacity")
        if sum(self.intent_counts.values()) != self.counts.generated:
            raise ValueError("intent counts must sum to generated requests")
        if sum(self.persona_counts.values()) != self.counts.generated:
            raise ValueError("persona counts must sum to generated requests")
        if self.staffing_mode is StaffingMode.FIXED and any(
            snapshot.busy_agents > snapshot.available_agents
            for snapshot in self.snapshots
        ):
            raise ValueError("snapshot busy-agent count cannot exceed available agents")
        return self


class ScenarioComparison(DomainModel):
    scenario_name: str
    generated_requests: int = Field(ge=0)
    average_waiting_time: float | None = Field(default=None, ge=0)
    sla: float | None = Field(default=None, ge=0, le=1)
    abandonment_rate: float = Field(ge=0, le=1)
    occupancy: float = Field(ge=0, le=1)
