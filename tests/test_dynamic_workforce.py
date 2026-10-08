from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import load_support_profile
from callverse.domain import CustomerPersona, RequestIntent
from callverse.scenarios import get_scenario
from callverse.simulation import (
    StaffingMode,
    StaffingSchedule,
    StaffingSlot,
    run_scheduled_simulation,
    run_simulation,
)
from callverse.simulation.policies import SimulationPolicy, TriangularMinutes
from callverse.workforce.dynamic import (
    FAIR_BASELINE_AGENTS,
    FAIR_EXPERIMENT_SEED,
    FAIR_HORIZON_MINUTES,
    SUPPORT_PROFILE_PATH,
    WorkforceComparisonOutcome,
    constant_staffing_schedule,
    fair_timeline_rows,
    load_current_forecast,
    run_fair_workforce_comparison,
    schedule_from_workforce_plan,
    staffing_change_events,
)
from callverse.workforce.manager import (
    build_workforce_plan,
    default_workforce_config,
)


@pytest.fixture(scope="module")
def forecast():
    return load_current_forecast()


@pytest.fixture(scope="module")
def workforce_plan(forecast):
    return build_workforce_plan(forecast, default_workforce_config())


@pytest.fixture(scope="module")
def fair_comparison(forecast):
    return run_fair_workforce_comparison(forecast)


def _schedule(values: tuple[int, ...], slot_minutes: float) -> StaffingSchedule:
    return StaffingSchedule(
        slots=tuple(
            StaffingSlot(
                start_minute=index * slot_minutes,
                end_minute=(index + 1) * slot_minutes,
                advisors=advisors,
            )
            for index, advisors in enumerate(values)
        ),
        slot_minutes=slot_minutes,
        horizon_minutes=len(values) * slot_minutes,
        source="Focused dynamic-capacity test",
        provenance=("Predefined test schedule",),
    )


def _capacity_test_policy() -> SimulationPolicy:
    handling = {
        intent: TriangularMinutes(20, 20, 20) for intent in RequestIntent
    }
    patience = {
        persona: TriangularMinutes(100, 100, 100)
        for persona in CustomerPersona
    }
    return SimulationPolicy(
        base_arrival_rate_per_minute=3,
        snapshot_interval_minutes=1,
        max_event_records=1_000,
        handling_times=handling,
        patience_times=patience,
    )


def _capacity_test_scenario():
    return get_scenario("normal_day").model_copy(
        update={
            "name": "scheduled_capacity_test",
            "simulation_duration": 40,
            "simulation_start_minute_of_day": 0,
            "random_seed": 77,
            "available_agents": 5,
        }
    )


def test_schedule_validation_and_exact_agent_hour_accounting():
    fixed = constant_staffing_schedule(
        advisors=2,
        horizon_minutes=FAIR_HORIZON_MINUTES,
    )
    mixed = _schedule((1, 2, 3, 4), 30)

    assert fixed.total_agent_hours == 48
    assert fixed.minimum_advisors == fixed.maximum_advisors == 2
    assert fixed.average_advisors == 2
    assert fixed.staffing_changes == 0
    assert mixed.total_agent_hours == 5
    assert mixed.minimum_advisors == 1
    assert mixed.maximum_advisors == 4
    assert mixed.average_advisors == 2.5
    assert mixed.staffing_changes == 3

    with pytest.raises(ValidationError, match="begin at simulation minute zero"):
        StaffingSchedule(
            slots=(StaffingSlot(start_minute=30, end_minute=60, advisors=2),),
            slot_minutes=30,
            horizon_minutes=60,
            source="invalid",
            provenance=("test",),
        )
    with pytest.raises(ValidationError, match="continuous and non-overlapping"):
        StaffingSchedule(
            slots=(
                StaffingSlot(start_minute=0, end_minute=30, advisors=2),
                StaffingSlot(start_minute=45, end_minute=75, advisors=2),
            ),
            slot_minutes=30,
            horizon_minutes=75,
            source="invalid",
            provenance=("test",),
        )
    with pytest.raises(ValidationError):
        StaffingSlot(start_minute=0, end_minute=30, advisors=0)
    with pytest.raises(ValueError, match="horizon must match"):
        run_scheduled_simulation(
            _capacity_test_scenario(),
            constant_staffing_schedule(advisors=2, horizon_minutes=30),
            policy=_capacity_test_policy(),
        )


def test_current_workforce_plan_import_and_provenance(forecast, workforce_plan):
    schedule = schedule_from_workforce_plan(workforce_plan, forecast)

    assert len(schedule.slots) == 48
    assert schedule.slot_minutes == 30
    assert schedule.horizon_minutes == 1440
    assert schedule.minimum_advisors == 1
    assert schedule.maximum_advisors == 5
    assert schedule.average_advisors == pytest.approx(1.7708333333333333)
    assert schedule.total_agent_hours == 42.5
    assert schedule.staffing_changes == 6
    assert schedule.source == "Forecast -> Erlang-C Workforce Manager"
    assert "lightgbm_poisson" in " ".join(schedule.provenance)
    assert "No PPO" in " ".join(schedule.provenance)


def test_workforce_plan_rejects_timestamp_misalignment(forecast, workforce_plan):
    shifted = forecast.model_copy(
        update={
            "points": tuple(
                point.model_copy(
                    update={"timestamp": point.timestamp + timedelta(minutes=30)}
                )
                for point in forecast.points
            )
        }
    )
    with pytest.raises(ValueError, match="timestamps must align exactly"):
        schedule_from_workforce_plan(workforce_plan, shifted)


def test_fixed_staff_shortage_regression_remains_exact():
    scenario = get_scenario("staff_shortage")
    policy = build_calibrated_policy(load_support_profile(SUPPORT_PROFILE_PATH))
    result = run_simulation(scenario, seed=404, policy=policy)

    assert result.staffing_mode is StaffingMode.FIXED
    assert result.counts.generated == 394
    assert result.counts.completed == 289
    assert result.counts.abandoned == 97
    assert result.kpis.sla == pytest.approx(0.6472602739726028)
    assert result.kpis.abandonment_rate == pytest.approx(0.24619289340101522)
    assert result.kpis.average_waiting_time == pytest.approx(2.112993469879009)
    assert result.kpis.occupancy == pytest.approx(0.850466403883364)
    assert result.snapshots[-1].queue_size == 5


def test_constant_schedule_is_exactly_equivalent_to_fixed_staffing():
    scenario = get_scenario("staff_shortage")
    policy = build_calibrated_policy(load_support_profile(SUPPORT_PROFILE_PATH))
    schedule = constant_staffing_schedule(
        advisors=3,
        horizon_minutes=scenario.simulation_duration,
    )
    fixed = run_simulation(scenario, seed=404, policy=policy)
    scheduled = run_scheduled_simulation(
        scenario, schedule, seed=404, policy=policy
    )

    assert fixed.counts == scheduled.counts
    assert fixed.kpis == scheduled.kpis
    assert fixed.snapshots == scheduled.snapshots
    assert fixed.request_records == scheduled.request_records
    assert fixed.busy_advisor_minutes == scheduled.busy_advisor_minutes
    assert fixed.capacity_advisor_minutes == scheduled.capacity_advisor_minutes


def test_capacity_reduction_preserves_active_services_without_preemption():
    schedule = _schedule((5, 2, 2, 2, 2, 2, 2, 2), 5)
    result = run_scheduled_simulation(
        _capacity_test_scenario(),
        schedule,
        seed=77,
        policy=_capacity_test_policy(),
    )
    boundary = next(
        snapshot for snapshot in result.snapshots if snapshot.simulation_time == 5
    )
    carried = [
        record
        for record in result.request_records
        if record.service_start is not None
        and record.service_start < 5
        and record.end_time is not None
        and record.end_time > 5
    ]

    assert len(carried) == 5
    assert boundary.available_agents == 2
    assert boundary.busy_agents == 5
    assert boundary.free_agents == 0
    assert boundary.overhang_busy_agents == 3
    assert all(
        record.end_time - record.service_start == pytest.approx(20)
        for record in carried
    )
    post_reduction_starts = sorted(
        record.service_start
        for record in result.request_records
        if record.service_start is not None and record.service_start >= 5
    )
    carried_completions = sorted(record.end_time for record in carried)
    assert post_reduction_starts[0] == carried_completions[3]


def test_capacity_increase_releases_waiting_contacts_at_boundary():
    schedule = _schedule((2, 5, 5, 5, 5, 5, 5, 5), 5)
    result = run_scheduled_simulation(
        _capacity_test_scenario(),
        schedule,
        seed=77,
        policy=_capacity_test_policy(),
    )
    boundary = next(
        snapshot for snapshot in result.snapshots if snapshot.simulation_time == 5
    )
    released = [
        record
        for record in result.request_records
        if record.arrival_time < 5 and record.service_start == 5
    ]

    assert len(released) == 3
    assert boundary.available_agents == 5
    assert boundary.busy_agents == 5


def test_main_fair_comparison_uses_matched_demand_and_actual_results(fair_comparison):
    baseline = fair_comparison.baseline_result
    callverse = fair_comparison.callverse_result

    assert fair_comparison.seed == FAIR_EXPERIMENT_SEED == 404
    assert baseline.counts.generated == callverse.counts.generated == 248
    assert baseline.counts.completed == 174
    assert callverse.counts.completed == 228
    assert baseline.counts.abandoned == 74
    assert callverse.counts.abandoned == 20
    assert baseline.kpis.sla == pytest.approx(0.6091954022988506)
    assert callverse.kpis.sla == pytest.approx(0.9078947368421053)
    assert baseline.kpis.average_waiting_time == pytest.approx(2.6645230206418375)
    assert callverse.kpis.average_waiting_time == pytest.approx(0.5363577738102044)
    assert baseline.kpis.occupancy == pytest.approx(0.24814401948842232)
    assert callverse.kpis.occupancy == pytest.approx(0.3696561994731159)
    assert baseline.snapshots[-1].queue_size == 0
    assert callverse.snapshots[-1].queue_size == 0
    assert fair_comparison.baseline_schedule.total_agent_hours == 48
    assert fair_comparison.callverse_schedule.total_agent_hours == 42.5
    assert fair_comparison.resource_budget_delta_hours == -5.5
    assert fair_comparison.fairness_satisfied
    assert (
        fair_comparison.outcome
        is WorkforceComparisonOutcome.IMPROVED_LOWER_BUDGET
    )

    assert len(baseline.request_records) == len(callverse.request_records) == 248
    for fixed, dynamic in zip(
        baseline.request_records, callverse.request_records, strict=True
    ):
        assert (
            fixed.arrival_time,
            fixed.intent,
            fixed.persona,
            fixed.patience_minutes,
            fixed.handling_minutes,
        ) == (
            dynamic.arrival_time,
            dynamic.intent,
            dynamic.persona,
            dynamic.patience_minutes,
            dynamic.handling_minutes,
        )


def test_fair_comparison_executes_both_real_simulator_paths_once(
    forecast, monkeypatch
):
    from callverse.workforce import dynamic

    fixed_runner = dynamic.run_simulation
    scheduled_runner = dynamic.run_scheduled_simulation
    calls = {"fixed": 0, "scheduled": 0}

    def count_fixed(*args, **kwargs):
        calls["fixed"] += 1
        return fixed_runner(*args, **kwargs)

    def count_scheduled(*args, **kwargs):
        calls["scheduled"] += 1
        return scheduled_runner(*args, **kwargs)

    monkeypatch.setattr(dynamic, "run_simulation", count_fixed)
    monkeypatch.setattr(dynamic, "run_scheduled_simulation", count_scheduled)
    comparison = dynamic.run_fair_workforce_comparison(forecast)

    assert calls == {"fixed": 1, "scheduled": 1}
    assert comparison.baseline_result.staffing_mode is StaffingMode.FIXED
    assert comparison.callverse_result.staffing_mode is StaffingMode.SCHEDULED


def test_staffing_events_and_timeline_are_real_and_aligned(
    forecast, workforce_plan, fair_comparison
):
    events = staffing_change_events(
        schedule_from_workforce_plan(workforce_plan, forecast), forecast
    )
    rows = fair_timeline_rows(fair_comparison, forecast)

    assert len(events) == 6
    assert [event.timestamp.strftime("%H:%M") for event in events] == [
        "17:30",
        "18:00",
        "18:30",
        "19:00",
        "20:00",
        "23:30",
    ]
    assert all("CallVerse staffing plan changes" in event.title for event in events)
    assert all("Manager" not in event.title for event in events)
    assert len(rows) == 48
    assert [row["simulated_clock"] for row in rows] == [
        point.timestamp.strftime("%H:%M") for point in forecast.points
    ]
    assert rows[0]["baseline_advisors"] == FAIR_BASELINE_AGENTS
    assert rows[38]["callverse_advisors"] == 5
    assert max(row["callverse_overhang"] for row in rows) == 1
    schedule = fair_comparison.callverse_schedule
    for snapshot in fair_comparison.callverse_result.snapshots:
        slot_index = min(
            int(snapshot.simulation_time // schedule.slot_minutes),
            len(schedule.slots) - 1,
        )
        assert snapshot.available_agents == schedule.slots[slot_index].advisors
        assert snapshot.free_agents == max(
            snapshot.available_agents - snapshot.busy_agents, 0
        )
        assert snapshot.overhang_busy_agents == max(
            snapshot.busy_agents - snapshot.available_agents, 0
        )


def test_streamlit_fair_workforce_comparison_is_explicit_and_functional():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    visible = "\n".join(
        item.value
        for item in (*app.subheader, *app.caption, *app.info, *app.warning)
    )
    assert "FAIR WORKFORCE COMPARISON" in visible
    assert "WORKFORCE INTELLIGENCE" in visible
    assert "CAPACITY WHAT-IF" in visible
    assert "Nothing auto-runs" in visible

    next(
        button
        for button in app.button
        if button.label == "RUN FAIR WORKFORCE COMPARISON"
    ).click()
    app.run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Baseline agent-hours"] == "48.0"
    assert metrics["CallVerse agent-hours"] == "42.5"
    assert metrics["Budget difference"] == "-5.5 h"
    assert any("RESOURCE BUDGET PASSES" in item.value for item in app.success)
