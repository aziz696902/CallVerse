"""Typed contracts for the analytical CallVerse Workforce Manager."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from pydantic import Field, model_validator

from callverse.domain import DomainModel


class WorkforceConfig(DomainModel):
    mean_aht_minutes: float = Field(gt=0)
    sla_wait_threshold_minutes: float = Field(ge=0)
    target_service_level: float = Field(ge=0, le=1)
    max_occupancy: float = Field(gt=0, lt=1)
    min_agents: int = Field(default=1, ge=0)
    max_agents: int = Field(default=50, gt=0)
    forecast_buffer_percent: float = Field(default=10.0, ge=0, le=100)
    reduction_hold_intervals: int = Field(default=2, ge=1, le=48)
    cost_per_agent_hour: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_agent_limits(self):
        if self.min_agents > self.max_agents:
            raise ValueError("minimum agents cannot exceed maximum agents")
        return self

    @property
    def service_rate_per_hour(self) -> float:
        return 60 / self.mean_aht_minutes


class ErlangCResult(DomainModel):
    arrival_rate_per_hour: float = Field(ge=0)
    service_rate_per_hour_per_agent: float = Field(gt=0)
    agents: int = Field(ge=0)
    offered_load: float = Field(ge=0)
    utilization: float = Field(ge=0)
    probability_wait: float = Field(ge=0, le=1)
    expected_wait_minutes: float | None = Field(default=None, ge=0)
    service_level: float = Field(ge=0, le=1)
    stable: bool


class StaffingSearchResult(DomainModel):
    agents: int = Field(ge=0)
    metrics: ErlangCResult
    target_met: bool
    capacity_shortfall: bool


class StaffingRecommendationPoint(DomainModel):
    timestamp: datetime
    forecast_contacts: float = Field(ge=0)
    buffered_contacts: float = Field(ge=0)
    arrival_rate_per_hour: float = Field(ge=0)
    raw_required_agents: int = Field(ge=0)
    recommended_agents: int = Field(ge=0)
    offered_load: float = Field(ge=0)
    predicted_utilization: float = Field(ge=0)
    predicted_service_level: float = Field(ge=0, le=1)
    predicted_wait_minutes: float | None = Field(default=None, ge=0)
    target_met: bool
    capacity_shortfall: bool
    smoothing_applied: bool
    explanation: str = Field(min_length=1)


class WorkforcePlan(DomainModel):
    forecast_origin: datetime
    interval_minutes: int = Field(gt=0)
    points: tuple[StaffingRecommendationPoint, ...]
    config: WorkforceConfig
    method: str = "Erlang-C analytical staffing baseline"
    version: str = "callverse-workforce-v1"

    @model_validator(mode="after")
    def validate_plan(self):
        if len(self.points) != 48:
            raise ValueError("workforce plan must contain exactly 48 half-hour points")
        if self.interval_minutes != 30:
            raise ValueError("workforce plan requires 30-minute intervals")
        for previous, current in pairwise(self.points):
            if (current.timestamp - previous.timestamp).total_seconds() != 1800:
                raise ValueError("workforce timestamps must progress by 30 minutes")
        return self


class WorkforceSummary(DomainModel):
    minimum_agents: int = Field(ge=0)
    maximum_agents: int = Field(ge=0)
    average_agents: float = Field(ge=0)
    total_agent_hours: float = Field(ge=0)
    staffing_changes: int = Field(ge=0)
    peak_staffing_timestamps: tuple[datetime, ...]
    capacity_shortfall_intervals: int = Field(ge=0)
    target_attainment_intervals: int = Field(ge=0)
    estimated_staffing_cost: float | None = Field(default=None, ge=0)


class StrategySummary(DomainModel):
    strategy: str
    baseline_definition: str
    total_agent_hours: float = Field(ge=0)
    average_agents: float = Field(ge=0)
    peak_agents: int = Field(ge=0)
    target_attainment_intervals: int = Field(ge=0)
    average_utilization: float = Field(ge=0)
    average_service_level: float = Field(ge=0, le=1)
    capacity_shortfall_intervals: int = Field(ge=0)


class StaffingStrategyComparison(DomainModel):
    fixed: StrategySummary
    erlang_c: StrategySummary

