"""Forecast-to-staffing application service."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from statistics import fmean

from callverse.calibration.profiles import load_support_profile
from callverse.forecasting.contracts import DemandForecast
from callverse.simulation import DEFAULT_POLICY

from .erlang_c import evaluate_erlang_c, minimum_agents
from .models import (
    StaffingRecommendationPoint,
    StaffingStrategyComparison,
    StrategySummary,
    WorkforceConfig,
    WorkforcePlan,
    WorkforceSummary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_PROFILE_PATH = PROJECT_ROOT / "data/processed/calibration/support_center_profile.json"


def default_workforce_config(**updates) -> WorkforceConfig:
    profile = load_support_profile(SUPPORT_PROFILE_PATH)
    mean_aht = profile.service_time_minutes.mean
    if mean_aht is None:
        raise ValueError("calibrated mean service time is unavailable")
    values = {
        "mean_aht_minutes": mean_aht,
        "sla_wait_threshold_minutes": DEFAULT_POLICY.sla_target_minutes,
        "target_service_level": 0.80,
        "max_occupancy": 0.85,
        "min_agents": 1,
        "max_agents": 50,
        "forecast_buffer_percent": 10.0,
        "reduction_hold_intervals": 2,
        "cost_per_agent_hour": None,
    }
    values.update(updates)
    return WorkforceConfig(**values)


def _smooth_requirements(raw_requirements: list[int], hold_intervals: int) -> list[int]:
    if not raw_requirements:
        return []
    schedule = [raw_requirements[0]]
    lower_streak = 0
    for raw in raw_requirements[1:]:
        current = schedule[-1]
        if raw >= current:
            schedule.append(raw)
            lower_streak = 0
        else:
            lower_streak += 1
            if lower_streak >= hold_intervals:
                schedule.append(raw)
                lower_streak = 0
            else:
                schedule.append(current)
    return schedule


def build_workforce_plan(forecast: DemandForecast, config: WorkforceConfig) -> WorkforcePlan:
    if forecast.interval_minutes != 30 or len(forecast.points) != 48:
        raise ValueError("Workforce Manager V1 requires a 48-point half-hour forecast")
    service_rate = config.service_rate_per_hour
    searches = []
    for point in forecast.points:
        buffered = point.predicted_contacts * (1 + config.forecast_buffer_percent / 100)
        arrival_rate = buffered / (forecast.interval_minutes / 60)
        searches.append(
            (
                point,
                buffered,
                arrival_rate,
                minimum_agents(
                    arrival_rate_per_hour=arrival_rate,
                    service_rate_per_hour_per_agent=service_rate,
                    sla_wait_threshold_minutes=config.sla_wait_threshold_minutes,
                    target_service_level=config.target_service_level,
                    max_occupancy=config.max_occupancy,
                    min_agents=config.min_agents,
                    max_agents=config.max_agents,
                ),
            )
        )
    raw_requirements = [search.agents for _, _, _, search in searches]
    operational = _smooth_requirements(raw_requirements, config.reduction_hold_intervals)
    recommendations = []
    for (point, buffered, arrival_rate, search), scheduled_agents in zip(
        searches, operational, strict=True
    ):
        metrics = evaluate_erlang_c(
            arrival_rate,
            service_rate,
            scheduled_agents,
            config.sla_wait_threshold_minutes,
        )
        target_met = (
            metrics.stable
            and metrics.service_level >= config.target_service_level
            and metrics.utilization <= config.max_occupancy
        )
        shortfall = search.capacity_shortfall and scheduled_agents == config.max_agents
        if shortfall:
            explanation = (
                f"No staffing level through {config.max_agents} agents satisfies both targets; "
                "the maximum evaluated capacity is shown."
            )
        else:
            explanation = (
                f"{scheduled_agents} agents keep modeled utilization at "
                f"{metrics.utilization:.1%} and modeled service level at "
                f"{metrics.service_level:.1%} for this buffered forecast."
            )
        recommendations.append(
            StaffingRecommendationPoint(
                timestamp=point.timestamp,
                forecast_contacts=point.predicted_contacts,
                buffered_contacts=buffered,
                arrival_rate_per_hour=arrival_rate,
                raw_required_agents=search.agents,
                recommended_agents=scheduled_agents,
                offered_load=metrics.offered_load,
                predicted_utilization=metrics.utilization,
                predicted_service_level=metrics.service_level,
                predicted_wait_minutes=metrics.expected_wait_minutes,
                target_met=target_met,
                capacity_shortfall=shortfall,
                smoothing_applied=scheduled_agents != search.agents,
                explanation=explanation,
            )
        )
    return WorkforcePlan(
        forecast_origin=forecast.forecast_origin,
        interval_minutes=forecast.interval_minutes,
        points=tuple(recommendations),
        config=config,
    )


def summarize_plan(plan: WorkforcePlan) -> WorkforceSummary:
    agents = [point.recommended_agents for point in plan.points]
    peak = max(agents)
    agent_hours = sum(agents) * plan.interval_minutes / 60
    return WorkforceSummary(
        minimum_agents=min(agents),
        maximum_agents=peak,
        average_agents=fmean(agents),
        total_agent_hours=agent_hours,
        staffing_changes=sum(left != right for left, right in pairwise(agents)),
        peak_staffing_timestamps=tuple(
            point.timestamp for point in plan.points if point.recommended_agents == peak
        ),
        capacity_shortfall_intervals=sum(point.capacity_shortfall for point in plan.points),
        target_attainment_intervals=sum(point.target_met for point in plan.points),
        estimated_staffing_cost=(
            None
            if plan.config.cost_per_agent_hour is None
            else agent_hours * plan.config.cost_per_agent_hour
        ),
    )


def compare_staffing_strategies(
    plan: WorkforcePlan,
    fixed_agents: int | None = None,
) -> StaffingStrategyComparison:
    summary = summarize_plan(plan)
    fixed = fixed_agents if fixed_agents is not None else max(1, round(summary.average_agents))
    fixed_metrics = [
        evaluate_erlang_c(
            point.arrival_rate_per_hour,
            plan.config.service_rate_per_hour,
            fixed,
            plan.config.sla_wait_threshold_minutes,
        )
        for point in plan.points
    ]
    fixed_attainment = [
        metric.stable
        and metric.service_level >= plan.config.target_service_level
        and metric.utilization <= plan.config.max_occupancy
        for metric in fixed_metrics
    ]
    variable_metrics = [
        evaluate_erlang_c(
            point.arrival_rate_per_hour,
            plan.config.service_rate_per_hour,
            point.recommended_agents,
            plan.config.sla_wait_threshold_minutes,
        )
        for point in plan.points
    ]
    return StaffingStrategyComparison(
        fixed=StrategySummary(
            strategy="fixed_staffing",
            baseline_definition=(
                f"{fixed} agents in every interval; rounded average Erlang-C plan requirement"
                if fixed_agents is None
                else f"manager-selected fixed staffing of {fixed} agents"
            ),
            total_agent_hours=fixed * 24,
            average_agents=float(fixed),
            peak_agents=fixed,
            target_attainment_intervals=sum(fixed_attainment),
            average_utilization=fmean(metric.utilization for metric in fixed_metrics),
            average_service_level=fmean(metric.service_level for metric in fixed_metrics),
            capacity_shortfall_intervals=sum(not attained for attained in fixed_attainment),
        ),
        erlang_c=StrategySummary(
            strategy="erlang_c_operationalized",
            baseline_definition="minimum analytical requirement with deterministic reduction hold",
            total_agent_hours=summary.total_agent_hours,
            average_agents=summary.average_agents,
            peak_agents=summary.maximum_agents,
            target_attainment_intervals=summary.target_attainment_intervals,
            average_utilization=fmean(metric.utilization for metric in variable_metrics),
            average_service_level=fmean(metric.service_level for metric in variable_metrics),
            capacity_shortfall_intervals=summary.capacity_shortfall_intervals,
        ),
    )


def workforce_chart_rows(plan: WorkforcePlan) -> list[dict[str, object]]:
    return [
        {
            "timestamp": point.timestamp,
            "forecast_contacts": point.forecast_contacts,
            "raw_agents": point.raw_required_agents,
            "recommended_agents": point.recommended_agents,
        }
        for point in plan.points
    ]
