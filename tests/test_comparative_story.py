from __future__ import annotations

from itertools import pairwise

import pytest
from streamlit.testing.v1 import AppTest

from callverse.dashboard.comparative_replay import (
    ASSISTED_LABEL,
    BASELINE_LABEL,
    DecisionSource,
    build_comparative_replay,
    create_comparison_configuration,
)
from callverse.dashboard.comparative_story import (
    LARGE_CENTER_ASSISTED_AGENTS,
    LARGE_CENTER_BASELINE_AGENTS,
    LARGE_CENTER_DEMAND_MULTIPLIER,
    ReplayEventType,
    build_final_manager_story,
    current_frame_differences,
    generate_replay_events,
    large_center_configuration,
    queue_trajectory_rows,
)
from callverse.dashboard.decision_guidance import (
    DecisionOutcome,
    build_decision_narrative,
)
from callverse.dashboard.playback import (
    SUPPORTED_SPEEDS,
    advance,
    advisor_markers,
    current_frame,
    initial_playback_state,
    pause,
    play,
    queue_markers,
    restart,
    scrub,
    set_speed,
)
from callverse.dashboard.scenario_guidance import (
    CenterStatus,
    interpret_simulation_result,
)


def official_configuration():
    return create_comparison_configuration(
        "staff_shortage",
        seed=404,
        duration_minutes=480,
        demand_multiplier=1.15,
        baseline_agents=3,
        assisted_agents=5,
        policy_mode="calibrated",
        decision_source=DecisionSource.CAPACITY_WHAT_IF,
    )


@pytest.fixture(scope="module")
def official_replay():
    return build_comparative_replay(official_configuration())


@pytest.fixture(scope="module")
def large_replay():
    return build_comparative_replay(large_center_configuration())


def test_events_are_deterministic_real_first_occurrences_without_intervention(
    official_replay,
):
    events = generate_replay_events(official_replay)

    assert events == generate_replay_events(official_replay)
    assert len({(event.side, event.event_type) for event in events}) == len(events)
    assert all(
        event.simulation_time == official_replay.frames[event.frame_index].simulation_minute
        and event.simulated_clock == official_replay.frames[event.frame_index].simulated_clock
        for event in events
    )
    normalized_text = " ".join(
        f"{event.title} {event.description}" for event in events
    ).lower()
    assert "intervention" not in normalized_text
    assert "adds" not in normalized_text

    by_key = {(event.side, event.event_type): event for event in events}
    assert by_key[("baseline", ReplayEventType.QUEUE_EMERGES)].frame_index == 3
    assert by_key[("baseline", ReplayEventType.ALL_ADVISORS_BUSY)].frame_index == 3
    assert by_key[("baseline", ReplayEventType.PRESSURE_HIGH)].frame_index == 3
    assert by_key[("baseline", ReplayEventType.PRESSURE_SEVERE)].frame_index == 10
    assert by_key[("assisted", ReplayEventType.QUEUE_EMERGES)].frame_index == 10
    assert by_key[("comparison", ReplayEventType.SIMULATION_END)].frame_index == 32


@pytest.mark.parametrize(
    ("frame_index", "expected"),
    [
        (0, ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1))),
        (16, ((0, 0, 0, 0), (153, 192, 39, 1), (47, 8, -39, 1))),
        (32, ((5, 0, -5, 0), (289, 373, 84, 1), (97, 17, -80, 1))),
    ],
)
def test_current_frame_differences_use_correct_values_signs_and_semantics(
    official_replay, frame_index, expected
):
    differences = {
        difference.metric: difference
        for difference in current_frame_differences(official_replay.frames[frame_index])
    }

    queue, completed, abandoned = expected
    for metric, values in (
        ("Current queue", queue),
        ("Completed so far", completed),
        ("Abandoned so far", abandoned),
    ):
        difference = differences[metric]
        assert (
            difference.baseline,
            difference.assisted,
            difference.assisted_delta,
        ) == values[:3]
        expected_semantics = (
            "current snapshot" if values[3] == 0 else "cumulative from simulation start"
        )
        assert difference.semantics == expected_semantics
    assert differences["Busy advisors"].semantics == "current snapshot"


def test_queue_trajectory_uses_every_real_frame_without_interpolation(official_replay):
    rows = queue_trajectory_rows(official_replay)

    assert len(rows) == len(official_replay.frames) == 33
    assert rows == tuple(
        {
            "simulation_time": frame.simulation_minute,
            BASELINE_LABEL: frame.baseline.queue_size,
            ASSISTED_LABEL: frame.assisted.queue_size,
        }
        for frame in official_replay.frames
    )


def test_final_story_reuses_existing_decision_guidance(official_replay):
    story = build_final_manager_story(official_replay)

    assert story.narrative == build_decision_narrative(official_replay.comparison)
    assert story.narrative.outcome is DecisionOutcome.STRONGLY_IMPROVED
    assert story.staffing_delta == 2
    assert story.staffing_trade_off == "Additional staffing: +2 advisors"
    assert {
        "Improved SLA",
        "Lower abandonment",
        "Lower average wait",
        "Higher completed contacts",
        "Lower final backlog",
    } <= set(story.service_effects)
    assert "production optimum" in story.scientific_boundary


@pytest.mark.parametrize(
    ("scale", "expected_generated", "expected_completed", "expected_queues"),
    [
        (5, 2117, (1726, 2085), (1, 0)),
        (8, 3358, (2845, 3330), (4, 0)),
        (10, 4211, (3590, 4172), (4, 0)),
    ],
)
def test_predefined_large_center_grid_is_deterministic_and_aligned(
    scale, expected_generated, expected_completed, expected_queues
):
    configured = create_comparison_configuration(
        "staff_shortage",
        seed=404,
        duration_minutes=480,
        demand_multiplier=1.15 * scale,
        baseline_agents=3 * scale,
        assisted_agents=5 * scale,
        policy_mode="calibrated",
        decision_source=DecisionSource.SCALABILITY_DEMO,
    )
    replay = build_comparative_replay(configured)
    baseline = replay.comparison.before.result
    assisted = replay.comparison.after.result

    assert baseline.counts.generated == assisted.counts.generated == expected_generated
    assert (baseline.counts.completed, assisted.counts.completed) == expected_completed
    assert (
        baseline.snapshots[-1].queue_size,
        assisted.snapshots[-1].queue_size,
    ) == expected_queues
    assert replay.configuration.seed == 404
    assert replay.snapshot_cadence_minutes == 15
    assert len(replay.frames) == 33
    assert all(frame.baseline.busy_agents <= 3 * scale for frame in replay.frames)
    assert all(frame.assisted.busy_agents <= 5 * scale for frame in replay.frames)
    for side_name in ("baseline", "assisted"):
        sides = [getattr(frame, side_name) for frame in replay.frames]
        assert all(side.queue_size >= 0 for side in sides)
        assert all(
            current.completed_so_far >= previous.completed_so_far
            and current.abandoned_so_far >= previous.abandoned_so_far
            for previous, current in pairwise(sides)
        )


def test_selected_large_center_configuration_and_visual_caps(large_replay):
    configured = large_replay.configuration

    assert configured.scenario_name == "staff_shortage"
    assert configured.seed == 404
    assert configured.duration_minutes == 480
    assert configured.demand_multiplier == LARGE_CENTER_DEMAND_MULTIPLIER == 5.75
    assert configured.baseline_agents == LARGE_CENTER_BASELINE_AGENTS == 15
    assert configured.assisted_agents == LARGE_CENTER_ASSISTED_AGENTS == 25
    assert configured.decision_source is DecisionSource.SCALABILITY_DEMO
    assert queue_markers(25).hidden_count == 15
    assert advisor_markers(25, 0).hidden_count == 15


def test_normal_day_eight_to_ten_control_remains_healthy():
    replay = build_comparative_replay(
        create_comparison_configuration(
            "normal_day",
            seed=101,
            duration_minutes=480,
            demand_multiplier=1,
            baseline_agents=8,
            assisted_agents=10,
            policy_mode="calibrated",
        )
    )

    assert interpret_simulation_result(
        replay.comparison.before.result
    ).status is CenterStatus.HEALTHY
    assert interpret_simulation_result(
        replay.comparison.after.result
    ).status is CenterStatus.HEALTHY
    assert replay.comparison.before.result.counts.generated == 352
    assert replay.comparison.after.result.counts.generated == 352
    assert replay.comparison.before.result.kpis.sla == 1
    assert replay.comparison.after.result.kpis.sla == 1
    assert all(frame.baseline.queue_size == 0 for frame in replay.frames)
    assert all(frame.assisted.queue_size == 0 for frame in replay.frames)


def test_large_center_playback_reads_stored_frames_without_simulation(
    large_replay, monkeypatch
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("playback must not invoke the Digital Twin")

    monkeypatch.setattr("callverse.dashboard.view_models.run_simulation", fail_if_called)
    total_frames = len(large_replay.frames)
    for speed in SUPPORTED_SPEEDS:
        state = set_speed(initial_playback_state(total_frames), speed)
        state = play(state, total_frames)
        state = advance(state, total_frames)
        state = pause(state)
        state = scrub(state, 16, total_frames)
        assert current_frame(large_replay, state) == large_replay.frames[16]
        assert restart(state, total_frames).frame_index == 0


def test_streamlit_storytelling_and_large_center_setup_are_visible():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    labels = {button.label for button in app.button}
    assert "PREPARE CAPACITY WHAT-IF" in labels
    assert "PREPARE SAME-STAFF CONTROL" in labels
    assert "PREPARE LARGE CENTER STRESS TEST" in labels
    visible = "\n".join(
        item.value for item in (*app.markdown, *app.caption, *app.info, *app.warning)
    )
    assert "TESTED DECISION" in visible
    assert "Both sides receive identical seeded demand conditions" in visible
    assert "controlled simulation comparison" in visible

    next(
        button
        for button in app.button
        if button.label == "PREPARE LARGE CENTER STRESS TEST"
    ).click()
    app.run(timeout=30)
    inputs = {widget.label: widget.value for widget in app.number_input}
    assert inputs["Seed"] == 404
    assert inputs["Baseline advisors"] == 15
    assert inputs["CallVerse-assisted advisors"] == 25
    assert inputs["Comparison demand multiplier"] == 5.75
    assert any("Scalability demonstration only" in item.value for item in app.warning)

    next(button for button in app.button if button.label == "BUILD COMPARISON").click()
    app.run(timeout=30)
    assert not app.exception
    assert any(
        widget.label == "Synchronized simulated time" for widget in app.select_slider
    )
    assert any("+ 15 additional advisors" in item.value for item in app.caption)


def test_streamlit_final_frame_promotes_manager_summary():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    next(button for button in app.button if button.label == "BUILD COMPARISON").click()
    app.run(timeout=30)
    timeline = next(
        slider
        for slider in app.select_slider
        if slider.label == "Synchronized simulated time"
    )
    timeline.set_value(32)
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Replay complete" for item in app.success)
    assert any(item.value == "FINAL MANAGER SUMMARY" for item in app.subheader)
    assert any("Simulation ends" in item.value for item in app.info)
