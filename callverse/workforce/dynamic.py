"""Fair fixed-versus-scheduled workforce experiment over the Digital Twin."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from enum import Enum
from itertools import pairwise
from pathlib import Path

import pandas as pd
from pydantic import Field, model_validator

from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import load_support_profile
from callverse.domain import DomainModel
from callverse.forecasting.contracts import DemandForecast
from callverse.forecasting.service import forecast_next_24h
from callverse.scenarios import get_scenario
from callverse.simulation import (
    SimulationResult,
    StaffingSchedule,
    StaffingSlot,
    run_scheduled_simulation,
    run_simulation,
)

from .manager import build_workforce_plan, default_workforce_config
from .models import WorkforcePlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_SERIES_PATH = PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv"
FORECAST_ARTIFACT_PATH = PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"
FAIR_EXPERIMENT_SEED = 404
FAIR_BASELINE_AGENTS = 2
FAIR_HORIZON_MINUTES = 24 * 60
SUPPORT_PROFILE_PATH = PROJECT_ROOT / "data/processed/calibration/support_center_profile.json"


class WorkforceComparisonOutcome(str, Enum):
    IMPROVED_LOWER_BUDGET = "IMPROVED WITH LOWER RESOURCE BUDGET"
    IMPROVED_SAME_BUDGET = "BETTER ALLOCATION WITH SAME RESOURCE BUDGET"
    MIXED = "MIXED"
    WORSENED = "WORSENED"
    LITTLE_CHANGE = "LITTLE CHANGE"
    UNFAIR_BUDGET = "RESOURCE BUDGET NOT FAIR"


class StaffingChangeEvent(DomainModel):
    slot_index: int = Field(ge=1)
    simulation_minute: float = Field(gt=0)
    timestamp: datetime
    previous_advisors: int = Field(ge=1)
    advisors: int = Field(ge=1)
    title: str = Field(min_length=1)


class WorkforcePolicyComparison(DomainModel):
    demand_source: str = Field(min_length=1)
    forecast_origin: datetime
    forecast_total_contacts: float = Field(ge=0)
    seed: int = Field(ge=0)
    baseline_schedule: StaffingSchedule
    callverse_schedule: StaffingSchedule
    baseline_result: SimulationResult
    callverse_result: SimulationResult
    baseline_staffing_events: tuple[StaffingChangeEvent, ...]
    staffing_events: tuple[StaffingChangeEvent, ...]
    shared_workforce_pool: int = Field(ge=1)
    resource_budget_equal: bool
    fairness_satisfied: bool
    resource_budget_delta_hours: float
    outcome: WorkforceComparisonOutcome

    @model_validator(mode="after")
    def validate_matched_experiment(self) -> WorkforcePolicyComparison:
        if not (
            self.baseline_result.seed == self.callverse_result.seed == self.seed
        ):
            raise ValueError("workforce comparison seeds do not match")
        if self.baseline_result.counts.generated != self.callverse_result.counts.generated:
            raise ValueError("workforce comparison did not preserve realized demand")
        expected_delta = (
            self.callverse_schedule.total_agent_hours
            - self.baseline_schedule.total_agent_hours
        )
        if abs(expected_delta - self.resource_budget_delta_hours) > 1e-9:
            raise ValueError("workforce resource-budget delta is inconsistent")
        if self.fairness_satisfied != (expected_delta <= 1e-9):
            raise ValueError("workforce fairness flag is inconsistent")
        if self.resource_budget_equal != (abs(expected_delta) <= 1e-9):
            raise ValueError("same-resource budget flag is inconsistent")
        if max(
            self.baseline_schedule.maximum_advisors,
            self.callverse_schedule.maximum_advisors,
        ) > self.shared_workforce_pool:
            raise ValueError("a staffing schedule exceeds the shared workforce pool")
        return self


def load_current_forecast() -> DemandForecast:
    """Load the existing forecast artifacts without creating new model output."""

    history = pd.read_csv(
        FORECAST_SERIES_PATH, parse_dates=["timestamp"]
    ).set_index("timestamp")["contacts"]
    return forecast_next_24h(history, FORECAST_ARTIFACT_PATH)


def constant_staffing_schedule(
    *,
    advisors: int,
    horizon_minutes: float,
    slot_minutes: float = 30,
    source: str = "Fixed staffing baseline",
) -> StaffingSchedule:
    slot_count = horizon_minutes / slot_minutes
    if not slot_count.is_integer():
        raise ValueError("horizon must contain a whole number of staffing slots")
    return StaffingSchedule(
        slots=tuple(
            StaffingSlot(
                start_minute=index * slot_minutes,
                end_minute=(index + 1) * slot_minutes,
                advisors=advisors,
            )
            for index in range(int(slot_count))
        ),
        slot_minutes=slot_minutes,
        horizon_minutes=horizon_minutes,
        source=source,
        provenance=("Constant integer staffing specified before simulation",),
    )


def uniform_resource_schedule(
    *,
    slot_count: int,
    slot_minutes: float,
    total_agent_hours: float,
    max_advisors: int,
) -> StaffingSchedule:
    """Spread an exact staffing budget uniformly without demand information.

    Every slot receives the integer floor allocation. Remaining advisor-slot units
    are placed at the midpoint of equally sized partitions of the full horizon;
    integer-floor ties resolve toward the earlier slot.
    """

    if slot_count <= 0:
        raise ValueError("uniform schedule requires at least one slot")
    if slot_minutes <= 0:
        raise ValueError("uniform schedule interval must be positive")
    if max_advisors < 1:
        raise ValueError("uniform schedule workforce pool must be positive")
    advisor_slot_units = total_agent_hours * 60 / slot_minutes
    rounded_units = round(advisor_slot_units)
    if abs(advisor_slot_units - rounded_units) > 1e-9:
        raise ValueError("staffing budget must equal a whole advisor-slot unit")
    base_advisors, remainder = divmod(rounded_units, slot_count)
    if base_advisors < 1:
        raise ValueError("staffing budget must provide at least one advisor per slot")
    if base_advisors + bool(remainder) > max_advisors:
        raise ValueError("staffing budget exceeds the shared workforce pool")
    extra_slots = (
        {
            ((2 * index + 1) * slot_count) // (2 * remainder)
            for index in range(remainder)
        }
        if remainder
        else set()
    )
    if len(extra_slots) != remainder:
        raise ValueError("uniform placement did not produce unique slot positions")
    schedule = StaffingSchedule(
        slots=tuple(
            StaffingSlot(
                start_minute=index * slot_minutes,
                end_minute=(index + 1) * slot_minutes,
                advisors=base_advisors + (index in extra_slots),
            )
            for index in range(slot_count)
        ),
        slot_minutes=slot_minutes,
        horizon_minutes=slot_count * slot_minutes,
        source="Uniform demand-unaware resource baseline",
        provenance=(
            "No forecast, demand, or simulation outcome is accepted by this constructor",
            "Integer floor allocation in every slot",
            "Remainder placed at equal-partition midpoints across the full horizon",
            "Integer-floor ties resolve toward the earlier slot",
        ),
    )
    if abs(schedule.total_agent_hours - total_agent_hours) > 1e-9:
        raise ValueError("uniform schedule did not preserve the exact staffing budget")
    return schedule


def schedule_from_workforce_plan(
    plan: WorkforcePlan,
    forecast: DemandForecast,
) -> StaffingSchedule:
    """Map each forecast timestamp to the matching elapsed half-hour Twin slot."""

    if tuple(point.timestamp for point in plan.points) != tuple(
        point.timestamp for point in forecast.points
    ):
        raise ValueError("workforce and forecast timestamps must align exactly")
    if any(point.recommended_agents < 1 for point in plan.points):
        raise ValueError("dynamic Twin requires at least one advisor per slot")
    return StaffingSchedule(
        slots=tuple(
            StaffingSlot(
                start_minute=index * plan.interval_minutes,
                end_minute=(index + 1) * plan.interval_minutes,
                advisors=point.recommended_agents,
            )
            for index, point in enumerate(plan.points)
        ),
        slot_minutes=plan.interval_minutes,
        horizon_minutes=len(plan.points) * plan.interval_minutes,
        source="Forecast -> Erlang-C Workforce Manager",
        provenance=(
            f"Forecast model: {forecast.model_name} {forecast.model_version}",
            f"Forecast origin: {forecast.forecast_origin.isoformat()}",
            f"Workforce method: {plan.method} {plan.version}",
            "Default 80% SLA target, 85% occupancy cap, 10% forecast buffer",
            "No PPO and no post-simulation schedule tuning",
        ),
    )


def staffing_change_events(
    schedule: StaffingSchedule,
    forecast: DemandForecast,
    *,
    strategy_label: str = "CallVerse staffing plan",
) -> tuple[StaffingChangeEvent, ...]:
    events = []
    for index, (previous, current) in enumerate(
        pairwise(schedule.slots), start=1
    ):
        if previous.advisors == current.advisors:
            continue
        events.append(
            StaffingChangeEvent(
                slot_index=index,
                simulation_minute=current.start_minute,
                timestamp=forecast.points[index].timestamp,
                previous_advisors=previous.advisors,
                advisors=current.advisors,
                title=f"{strategy_label} changes to {current.advisors} advisors",
            )
        )
    return tuple(events)


def _comparison_outcome(
    baseline: SimulationResult,
    callverse: SimulationResult,
    *,
    fairness_satisfied: bool,
    resource_delta: float,
) -> WorkforceComparisonOutcome:
    if not fairness_satisfied:
        return WorkforceComparisonOutcome.UNFAIR_BUDGET
    service_directions = (
        (callverse.kpis.sla or 0) - (baseline.kpis.sla or 0),
        (baseline.kpis.abandonment_rate or 0)
        - (callverse.kpis.abandonment_rate or 0),
        (baseline.kpis.average_waiting_time or 0)
        - (callverse.kpis.average_waiting_time or 0),
    )
    tolerance = 1e-12
    improves = any(delta > tolerance for delta in service_directions)
    worsens = any(delta < -tolerance for delta in service_directions)
    if improves and worsens:
        return WorkforceComparisonOutcome.MIXED
    if worsens:
        return WorkforceComparisonOutcome.WORSENED
    if not improves:
        return WorkforceComparisonOutcome.LITTLE_CHANGE
    if resource_delta < -tolerance:
        return WorkforceComparisonOutcome.IMPROVED_LOWER_BUDGET
    return WorkforceComparisonOutcome.IMPROVED_SAME_BUDGET


def run_fair_workforce_comparison(
    forecast: DemandForecast,
    *,
    seed: int = FAIR_EXPERIMENT_SEED,
) -> WorkforcePolicyComparison:
    """Run the predefined, untuned fair staffing-policy experiment."""

    plan = build_workforce_plan(forecast, default_workforce_config())
    baseline_schedule = constant_staffing_schedule(
        advisors=FAIR_BASELINE_AGENTS,
        horizon_minutes=FAIR_HORIZON_MINUTES,
    )
    callverse_schedule = schedule_from_workforce_plan(plan, forecast)
    forecast_total = sum(point.predicted_contacts for point in forecast.points)
    calibrated_policy = build_calibrated_policy(
        load_support_profile(SUPPORT_PROFILE_PATH)
    )
    demand_multiplier = forecast_total / (
        calibrated_policy.base_arrival_rate_per_minute * FAIR_HORIZON_MINUTES
    )
    forecast_policy = replace(
        calibrated_policy,
        arrival_slot_multipliers=tuple(
            point.predicted_contacts for point in forecast.points
        ),
    )
    scenario = get_scenario("normal_day").model_copy(
        update={
            "name": "fair_workforce_forecast",
            "description": (
                "24-hour forecast-aligned fair fixed-versus-scheduled experiment"
            ),
            "simulation_duration": FAIR_HORIZON_MINUTES,
            "simulation_start_minute_of_day": 0,
            "random_seed": seed,
            "demand_multiplier": demand_multiplier,
            "available_agents": FAIR_BASELINE_AGENTS,
        }
    )
    baseline = run_simulation(scenario, seed=seed, policy=forecast_policy)
    callverse = run_scheduled_simulation(
        scenario,
        callverse_schedule,
        seed=seed,
        policy=forecast_policy,
    )
    resource_delta = (
        callverse_schedule.total_agent_hours
        - baseline_schedule.total_agent_hours
    )
    fairness = resource_delta <= 1e-9
    return WorkforcePolicyComparison(
        demand_source=(
            "Existing 48-slot LightGBM forecast, mapped directly to matching "
            "30-minute simulation intervals"
        ),
        forecast_origin=forecast.forecast_origin,
        forecast_total_contacts=forecast_total,
        seed=seed,
        baseline_schedule=baseline_schedule,
        callverse_schedule=callverse_schedule,
        baseline_result=baseline,
        callverse_result=callverse,
        baseline_staffing_events=(),
        staffing_events=staffing_change_events(callverse_schedule, forecast),
        shared_workforce_pool=max(
            baseline_schedule.maximum_advisors,
            callverse_schedule.maximum_advisors,
        ),
        resource_budget_equal=abs(resource_delta) <= 1e-9,
        fairness_satisfied=fairness,
        resource_budget_delta_hours=resource_delta,
        outcome=_comparison_outcome(
            baseline,
            callverse,
            fairness_satisfied=fairness,
            resource_delta=resource_delta,
        ),
    )


def run_same_resource_workforce_comparison(
    forecast: DemandForecast,
    *,
    seed: int = FAIR_EXPERIMENT_SEED,
) -> WorkforcePolicyComparison:
    """Run the predefined equal-budget allocation experiment without tuning."""

    plan = build_workforce_plan(forecast, default_workforce_config())
    callverse_schedule = schedule_from_workforce_plan(plan, forecast)
    shared_workforce_pool = callverse_schedule.maximum_advisors
    baseline_schedule = uniform_resource_schedule(
        slot_count=len(callverse_schedule.slots),
        slot_minutes=callverse_schedule.slot_minutes,
        total_agent_hours=callverse_schedule.total_agent_hours,
        max_advisors=shared_workforce_pool,
    )
    forecast_total = sum(point.predicted_contacts for point in forecast.points)
    calibrated_policy = build_calibrated_policy(
        load_support_profile(SUPPORT_PROFILE_PATH)
    )
    demand_multiplier = forecast_total / (
        calibrated_policy.base_arrival_rate_per_minute * FAIR_HORIZON_MINUTES
    )
    forecast_policy = replace(
        calibrated_policy,
        arrival_slot_multipliers=tuple(
            point.predicted_contacts for point in forecast.points
        ),
    )
    scenario = get_scenario("normal_day").model_copy(
        update={
            "name": "same_resource_workforce_forecast",
            "description": (
                "24-hour equal-budget uniform-versus-forecast allocation experiment"
            ),
            "simulation_duration": FAIR_HORIZON_MINUTES,
            "simulation_start_minute_of_day": 0,
            "random_seed": seed,
            "demand_multiplier": demand_multiplier,
            "available_agents": shared_workforce_pool,
        }
    )
    baseline = run_scheduled_simulation(
        scenario,
        baseline_schedule,
        seed=seed,
        policy=forecast_policy,
    )
    callverse = run_scheduled_simulation(
        scenario,
        callverse_schedule,
        seed=seed,
        policy=forecast_policy,
    )
    resource_delta = (
        callverse_schedule.total_agent_hours
        - baseline_schedule.total_agent_hours
    )
    budget_equal = abs(resource_delta) <= 1e-9
    return WorkforcePolicyComparison(
        demand_source=(
            "Existing 48-slot LightGBM forecast, mapped directly to matching "
            "30-minute simulation intervals"
        ),
        forecast_origin=forecast.forecast_origin,
        forecast_total_contacts=forecast_total,
        seed=seed,
        baseline_schedule=baseline_schedule,
        callverse_schedule=callverse_schedule,
        baseline_result=baseline,
        callverse_result=callverse,
        baseline_staffing_events=staffing_change_events(
            baseline_schedule,
            forecast,
            strategy_label="Uniform baseline scheduled capacity",
        ),
        staffing_events=staffing_change_events(callverse_schedule, forecast),
        shared_workforce_pool=shared_workforce_pool,
        resource_budget_equal=budget_equal,
        fairness_satisfied=budget_equal,
        resource_budget_delta_hours=resource_delta,
        outcome=_comparison_outcome(
            baseline,
            callverse,
            fairness_satisfied=budget_equal,
            resource_delta=resource_delta,
        ),
    )


def fair_timeline_rows(
    comparison: WorkforcePolicyComparison,
    forecast: DemandForecast,
) -> tuple[dict[str, object], ...]:
    baseline_snapshots = {
        snapshot.simulation_time: snapshot
        for snapshot in comparison.baseline_result.snapshots
    }
    callverse_snapshots = {
        snapshot.simulation_time: snapshot
        for snapshot in comparison.callverse_result.snapshots
    }
    arrivals = tuple(
        record.arrival_time for record in comparison.baseline_result.request_records
    )
    rows = []
    for index, (baseline_slot, callverse_slot, forecast_point) in enumerate(
        zip(
            comparison.baseline_schedule.slots,
            comparison.callverse_schedule.slots,
            forecast.points,
            strict=True,
        )
    ):
        baseline_snapshot = baseline_snapshots[baseline_slot.start_minute]
        callverse_snapshot = callverse_snapshots[callverse_slot.start_minute]
        realized = sum(
            baseline_slot.start_minute <= arrival < baseline_slot.end_minute
            for arrival in arrivals
        )
        rows.append(
            {
                "slot": index,
                "simulated_clock": forecast_point.timestamp.strftime("%H:%M"),
                "forecast_contacts": forecast_point.predicted_contacts,
                "realized_contacts": realized,
                "baseline_advisors": baseline_slot.advisors,
                "callverse_advisors": callverse_slot.advisors,
                "baseline_queue": baseline_snapshot.queue_size,
                "callverse_queue": callverse_snapshot.queue_size,
                "baseline_busy": baseline_snapshot.busy_agents,
                "callverse_busy": callverse_snapshot.busy_agents,
                "baseline_available": baseline_snapshot.available_agents,
                "callverse_available": callverse_snapshot.available_agents,
                "baseline_free": baseline_snapshot.free_agents,
                "callverse_free": callverse_snapshot.free_agents,
                "callverse_overhang": callverse_snapshot.overhang_busy_agents,
                "baseline_completed": baseline_snapshot.completed_count,
                "callverse_completed": callverse_snapshot.completed_count,
                "baseline_abandoned": baseline_snapshot.abandoned_count,
                "callverse_abandoned": callverse_snapshot.abandoned_count,
            }
        )
    return tuple(rows)
