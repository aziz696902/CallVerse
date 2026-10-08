from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from callverse.dashboard.comparative_replay import (
    ASSISTED_LABEL,
    BASELINE_LABEL,
    ComparativeReplay,
    DecisionSource,
    build_comparative_replay,
    comparison_matches_configuration,
    create_comparison_configuration,
    select_comparative_frame,
)
from callverse.dashboard.scenario_guidance import (
    CenterStatus,
    interpret_simulation_result,
)


def configuration(
    *, baseline_agents: int = 3, assisted_agents: int = 5
):
    return create_comparison_configuration(
        "staff_shortage",
        seed=404,
        duration_minutes=480,
        demand_multiplier=1.15,
        baseline_agents=baseline_agents,
        assisted_agents=assisted_agents,
        policy_mode="calibrated",
        decision_source=DecisionSource.RECOMMENDED_DEMO,
    )


@pytest.fixture(scope="module")
def official_replay() -> ComparativeReplay:
    return build_comparative_replay(configuration())


@pytest.fixture(scope="module")
def equal_replay() -> ComparativeReplay:
    return build_comparative_replay(
        configuration(baseline_agents=3, assisted_agents=3)
    )


def side_values(replay: ComparativeReplay, side: str) -> Iterator[dict[str, object]]:
    for frame in replay.frames:
        yield getattr(frame, side).model_dump(exclude={"label"})


def test_configuration_is_validated_and_immutable():
    configured = configuration()

    assert configured.scenario_name == "staff_shortage"
    assert configured.decision_source is DecisionSource.RECOMMENDED_DEMO
    with pytest.raises(ValidationError):
        configured.baseline_agents = 0
    with pytest.raises(ValidationError):
        configuration(baseline_agents=0)
    with pytest.raises(KeyError, match="unknown scenario"):
        create_comparison_configuration(
            "unknown",
            seed=1,
            duration_minutes=60,
            demand_multiplier=1,
            baseline_agents=1,
            assisted_agents=1,
            policy_mode="calibrated",
        )


def test_official_comparison_changes_only_staffing(official_replay):
    baseline = official_replay.comparison.before
    assisted = official_replay.comparison.after
    baseline_config = baseline.scenario.model_dump()
    assisted_config = assisted.scenario.model_dump()

    assert baseline.result.seed == assisted.result.seed == 404
    assert baseline.result.duration == assisted.result.duration == 480
    assert baseline.policy_mode == assisted.policy_mode == "calibrated"
    assert baseline.scenario.name == assisted.scenario.name == "staff_shortage"
    assert baseline.scenario.available_agents == 3
    assert assisted.scenario.available_agents == 5
    baseline_config.pop("available_agents")
    assisted_config.pop("available_agents")
    assert baseline_config == assisted_config


def test_timeline_is_exactly_aligned_and_complete(official_replay):
    baseline_times = tuple(
        snapshot.simulation_time
        for snapshot in official_replay.comparison.before.result.snapshots
    )
    assisted_times = tuple(
        snapshot.simulation_time
        for snapshot in official_replay.comparison.after.result.snapshots
    )

    assert baseline_times == assisted_times
    assert tuple(frame.simulation_minute for frame in official_replay.frames) == baseline_times
    assert len(official_replay.frames) == 33
    assert official_replay.snapshot_cadence_minutes == 15
    assert all(frame.frame_count == 33 for frame in official_replay.frames)
    assert [frame.index for frame in official_replay.frames] == list(range(33))
    assert all(frame.baseline.label == BASELINE_LABEL for frame in official_replay.frames)
    assert all(frame.assisted.label == ASSISTED_LABEL for frame in official_replay.frames)


def test_official_final_results_remain_at_trusted_values(official_replay):
    baseline = official_replay.comparison.before.result
    assisted = official_replay.comparison.after.result

    assert baseline.kpis.sla == pytest.approx(0.6473, abs=0.0001)
    assert baseline.kpis.abandonment_rate == pytest.approx(0.2462, abs=0.0001)
    assert baseline.kpis.average_waiting_time == pytest.approx(2.11, abs=0.01)
    assert baseline.kpis.occupancy == pytest.approx(0.8505, abs=0.0001)
    assert baseline.counts.completed == 289
    assert baseline.snapshots[-1].queue_size == 5

    assert assisted.kpis.sla == pytest.approx(0.9841, abs=0.0001)
    assert assisted.kpis.abandonment_rate == pytest.approx(0.0431, abs=0.0001)
    assert assisted.kpis.average_waiting_time == pytest.approx(0.20, abs=0.01)
    assert assisted.kpis.occupancy == pytest.approx(0.6237, abs=0.0001)
    assert assisted.counts.completed == 373
    assert assisted.snapshots[-1].queue_size == 0


def test_frame_values_change_and_preserve_counter_and_agent_invariants(official_replay):
    for side_name in ("baseline", "assisted"):
        sides = [getattr(frame, side_name) for frame in official_replay.frames]
        completed = [side.completed_so_far for side in sides]
        abandoned = [side.abandoned_so_far for side in sides]
        queues = [side.queue_size for side in sides]

        assert len(set(completed)) > 1
        assert len(set(abandoned)) > 1
        assert len({side.busy_agents for side in sides}) > 1
        assert completed == sorted(completed)
        assert abandoned == sorted(abandoned)
        assert min((*completed, *abandoned, *queues)) >= 0
        assert all(
            side.busy_agents + side.free_agents == side.available_agents
            for side in sides
        )
        assert not hasattr(sides[0], "sla")
        assert not hasattr(sides[0], "occupancy")
        assert not hasattr(sides[0], "average_wait")

    baseline_queues = [frame.baseline.queue_size for frame in official_replay.frames]
    assert any(current > previous for previous, current in pairwise(baseline_queues))
    assert any(current < previous for previous, current in pairwise(baseline_queues))


def test_selecting_frames_does_not_rerun_simulation(monkeypatch):
    from callverse.dashboard import view_models

    original = view_models.run_simulation
    calls = 0

    def counted_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(view_models, "run_simulation", counted_run)
    replay = build_comparative_replay(configuration())

    assert calls == 2
    assert select_comparative_frame(replay, 0) != select_comparative_frame(replay, 16)
    assert select_comparative_frame(replay, 32).index == 32
    assert calls == 2


def test_equal_staffing_control_is_identical_frame_by_frame(equal_replay):
    assert equal_replay.comparison.before == equal_replay.comparison.after
    assert len(equal_replay.frames) == 33
    assert tuple(side_values(equal_replay, "baseline")) == tuple(
        side_values(equal_replay, "assisted")
    )


def test_healthy_control_remains_healthy_without_manufactured_difference():
    healthy = build_comparative_replay(
        create_comparison_configuration(
            "normal_day",
            seed=101,
            duration_minutes=480,
            demand_multiplier=1,
            baseline_agents=8,
            assisted_agents=8,
            policy_mode="calibrated",
        )
    )

    assert interpret_simulation_result(healthy.comparison.before.result).status is CenterStatus.HEALTHY
    assert interpret_simulation_result(healthy.comparison.after.result).status is CenterStatus.HEALTHY
    assert tuple(side_values(healthy, "baseline")) == tuple(
        side_values(healthy, "assisted")
    )


def test_changed_setup_marks_stored_comparison_stale(official_replay):
    changed = configuration(assisted_agents=6)
    assert comparison_matches_configuration(official_replay, configuration())
    assert not comparison_matches_configuration(official_replay, changed)


def test_streamlit_comparative_replay_empty_build_and_stale_states():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    visible = "\n".join(item.value for item in (*app.subheader, *app.info, *app.caption))
    assert "COMPARATIVE SIMULATION REPLAY" in visible
    assert "Build a synchronized comparison first" in visible

    next(button for button in app.button if button.label == "PREPARE OFFICIAL DEMO").click()
    app.run(timeout=30)
    inputs = {widget.label: widget.value for widget in app.number_input}
    assert inputs["Seed"] == 404
    assert inputs["Baseline advisors"] == 3
    assert inputs["CallVerse-assisted advisors"] == 5
    assert not any(
        widget.label == "Synchronized simulated time" for widget in app.select_slider
    )

    next(button for button in app.button if button.label == "BUILD COMPARISON").click()
    app.run(timeout=30)
    assert not app.exception
    assert any(
        widget.label == "Synchronized simulated time" for widget in app.select_slider
    )
    visible = "\n".join(
        item.value for item in (*app.markdown, *app.caption, *app.info, *app.warning)
    )
    assert BASELINE_LABEL in visible
    assert ASSISTED_LABEL in visible
    assert "Frame 1 of 33" in visible
    assert "same calibrated Digital Twin" in visible
    assert "not live production telemetry" in visible

    next(
        widget
        for widget in app.number_input
        if widget.label == "CallVerse-assisted advisors"
    ).set_value(6)
    app.run(timeout=30)
    assert not app.exception
    assert any("setup changed" in item.value for item in app.warning)
    assert not any(
        widget.label == "Synchronized simulated time" for widget in app.select_slider
    )
