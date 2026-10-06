from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from callverse.calibration.profiles import load_support_profile
from callverse.forecasting.contracts import DemandForecast, ForecastPoint
from callverse.forecasting.service import forecast_next_24h
from callverse.workforce.erlang_c import evaluate_erlang_c, minimum_agents, offered_load
from callverse.workforce.manager import (
    build_workforce_plan,
    compare_staffing_strategies,
    default_workforce_config,
    summarize_plan,
    workforce_chart_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def forecast_fixture(values: list[float] | None = None) -> DemandForecast:
    values = values or [10.0] * 48
    start = pd.Timestamp("2026-02-01 00:00")
    return DemandForecast(
        forecast_origin=(start - timedelta(minutes=30)).to_pydatetime(),
        horizon=48,
        interval_minutes=30,
        points=tuple(
            ForecastPoint(
                timestamp=(start + timedelta(minutes=30 * index)).to_pydatetime(),
                predicted_contacts=value,
            )
            for index, value in enumerate(values)
        ),
        model_name="fixture",
        model_version="test",
    )


def test_erlang_c_zero_demand_is_stable_and_immediate():
    result = evaluate_erlang_c(0, 10, 0, 2)
    assert result.stable
    assert result.utilization == 0
    assert result.probability_wait == 0
    assert result.expected_wait_minutes == 0
    assert result.service_level == 1


def test_offered_load_uses_consistent_hourly_units():
    assert offered_load(60, 20) == 3


def test_known_mm1_case_matches_closed_form_wait():
    result = evaluate_erlang_c(2, 3, 1, 2)
    assert result.probability_wait == pytest.approx(2 / 3)
    assert result.expected_wait_minutes == pytest.approx(40)


def test_unstable_rho_is_explicit_without_division_error():
    result = evaluate_erlang_c(20, 10, 2, 2)
    assert not result.stable
    assert result.utilization == 1
    assert result.expected_wait_minutes is None
    assert result.service_level == 0


@pytest.mark.parametrize("agents", [2, 3, 5, 25, 100, 500])
def test_erlang_probabilities_remain_bounded(agents: int):
    result = evaluate_erlang_c(10, 10, agents, 2)
    assert 0 <= result.probability_wait <= 1
    assert 0 <= result.service_level <= 1


def test_expected_wait_reduces_monotonically_with_agents():
    results = [evaluate_erlang_c(20, 10, agents, 2) for agents in range(3, 8)]
    waits = [result.expected_wait_minutes for result in results]
    assert waits == sorted(waits, reverse=True)


def test_service_level_improves_with_agents():
    results = [evaluate_erlang_c(20, 10, agents, 2) for agents in range(3, 8)]
    levels = [result.service_level for result in results]
    assert levels == sorted(levels)


def test_minimum_staffing_selects_first_feasible_agent_count():
    search = minimum_agents(
        arrival_rate_per_hour=30,
        service_rate_per_hour_per_agent=10,
        sla_wait_threshold_minutes=2,
        target_service_level=0.8,
        max_occupancy=0.85,
        min_agents=1,
        max_agents=20,
    )
    assert search.target_met
    assert not search.capacity_shortfall


def test_one_fewer_agent_fails_at_least_one_constraint():
    search = minimum_agents(
        arrival_rate_per_hour=30,
        service_rate_per_hour_per_agent=10,
        sla_wait_threshold_minutes=2,
        target_service_level=0.8,
        max_occupancy=0.85,
        min_agents=1,
        max_agents=20,
    )
    previous = evaluate_erlang_c(30, 10, search.agents - 1, 2)
    assert previous.service_level < 0.8 or previous.utilization > 0.85 or not previous.stable


def test_max_occupancy_can_drive_staffing_above_sla_minimum():
    loose = minimum_agents(
        arrival_rate_per_hour=40,
        service_rate_per_hour_per_agent=10,
        sla_wait_threshold_minutes=10,
        target_service_level=0.5,
        max_occupancy=0.95,
        min_agents=1,
        max_agents=20,
    )
    strict = minimum_agents(
        arrival_rate_per_hour=40,
        service_rate_per_hour_per_agent=10,
        sla_wait_threshold_minutes=10,
        target_service_level=0.5,
        max_occupancy=0.60,
        min_agents=1,
        max_agents=20,
    )
    assert strict.agents > loose.agents
    assert strict.metrics.utilization <= 0.60


def test_capacity_shortfall_returns_explicit_max_candidate():
    search = minimum_agents(
        arrival_rate_per_hour=1000,
        service_rate_per_hour_per_agent=10,
        sla_wait_threshold_minutes=2,
        target_service_level=0.8,
        max_occupancy=0.85,
        min_agents=1,
        max_agents=5,
    )
    assert search.agents == 5
    assert search.capacity_shortfall
    assert not search.target_met


def test_forecast_becomes_48_preserved_workforce_timestamps():
    forecast = forecast_fixture()
    plan = build_workforce_plan(forecast, default_workforce_config(forecast_buffer_percent=0))
    assert len(plan.points) == 48
    assert [point.timestamp for point in plan.points] == [point.timestamp for point in forecast.points]


def test_arrival_rate_converts_half_hour_contacts_to_hourly_rate():
    plan = build_workforce_plan(forecast_fixture(), default_workforce_config())
    assert plan.points[0].forecast_contacts == 10
    assert plan.points[0].buffered_contacts == 11
    assert plan.points[0].arrival_rate_per_hour == 22


def test_calibrated_aht_and_service_rate_source():
    config = default_workforce_config()
    profile = load_support_profile(
        PROJECT_ROOT / "data/processed/calibration/support_center_profile.json"
    )
    assert config.mean_aht_minutes == profile.service_time_minutes.mean
    assert config.service_rate_per_hour == pytest.approx(18.8565512747)
    assert config.sla_wait_threshold_minutes == 2


def test_agent_hour_summary_is_half_hour_weighted():
    plan = build_workforce_plan(forecast_fixture(), default_workforce_config())
    summary = summarize_plan(plan)
    assert summary.total_agent_hours == sum(
        point.recommended_agents for point in plan.points
    ) * 0.5


def test_fixed_baseline_defaults_to_rounded_plan_average():
    plan = build_workforce_plan(forecast_fixture(), default_workforce_config())
    summary = summarize_plan(plan)
    comparison = compare_staffing_strategies(plan)
    assert comparison.fixed.average_agents == max(1, round(summary.average_agents))
    assert comparison.fixed.total_agent_hours == comparison.fixed.average_agents * 24


def test_reduction_hold_smoothing_never_drops_immediately():
    values = [25.0, 5.0, 25.0, 5.0] * 12
    plan = build_workforce_plan(
        forecast_fixture(values),
        default_workforce_config(forecast_buffer_percent=0, reduction_hold_intervals=2),
    )
    assert any(point.smoothing_applied for point in plan.points)
    assert all(point.recommended_agents >= point.raw_required_agents for point in plan.points)


def test_workforce_plan_is_deterministic():
    forecast = forecast_fixture([float(index % 15) for index in range(48)])
    config = default_workforce_config()
    assert build_workforce_plan(forecast, config) == build_workforce_plan(forecast, config)


def test_forecast_buffer_is_explicit_and_non_decreasing_for_staffing():
    forecast = forecast_fixture([20.0] * 48)
    base = build_workforce_plan(forecast, default_workforce_config(forecast_buffer_percent=0))
    buffered = build_workforce_plan(
        forecast, default_workforce_config(forecast_buffer_percent=20)
    )
    assert buffered.points[0].buffered_contacts == 24
    assert buffered.points[0].recommended_agents >= base.points[0].recommended_agents


def test_workforce_dashboard_rows_keep_demand_and_staffing_separate():
    plan = build_workforce_plan(forecast_fixture(), default_workforce_config())
    rows = workforce_chart_rows(plan)
    assert len(rows) == 48
    assert set(rows[0]) == {
        "timestamp",
        "forecast_contacts",
        "raw_agents",
        "recommended_agents",
    }


def test_recommendation_explanation_is_deterministic_and_analytical():
    point = build_workforce_plan(forecast_fixture(), default_workforce_config()).points[0]
    assert "modeled utilization" in point.explanation
    assert "modeled service level" in point.explanation


def test_optional_cost_defaults_to_unavailable():
    summary = summarize_plan(build_workforce_plan(forecast_fixture(), default_workforce_config()))
    assert summary.estimated_staffing_cost is None


def test_optional_cost_uses_configured_agent_hour_rate_only():
    plan = build_workforce_plan(
        forecast_fixture(), default_workforce_config(cost_per_agent_hour=12.5)
    )
    summary = summarize_plan(plan)
    assert summary.estimated_staffing_cost == summary.total_agent_hours * 12.5


def test_actual_phase9_forecast_builds_a_plan():
    history = pd.read_csv(
        PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv",
        parse_dates=["timestamp"],
    ).set_index("timestamp")["contacts"]
    forecast = forecast_next_24h(
        history, PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"
    )
    plan = build_workforce_plan(forecast, default_workforce_config())
    assert len(plan.points) == 48
    assert summarize_plan(plan).capacity_shortfall_intervals == 0


def test_permanent_validation_artifact_uses_multiple_twin_seeds():
    report = json.loads(
        (PROJECT_ROOT / "data/processed/workforce/erlang_c_validation.json").read_text()
    )
    assert report["dynamic_staffing"] == "deferred"
    assert {case["case"] for case in report["cases"]} == {
        "low_demand",
        "normal_demand",
        "high_demand",
        "near_saturation",
    }
    assert all(case["digital_twin"]["replications"] == 5 for case in report["cases"])
    assert all(len(case["digital_twin"]["seeds"]) == 5 for case in report["cases"])
