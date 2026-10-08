from __future__ import annotations

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from callverse.dashboard.comparative_replay import (
    build_comparative_replay,
    create_comparison_configuration,
)
from callverse.dashboard.playback import (
    SUPPORTED_SPEEDS,
    PlaybackState,
    advance,
    advisor_markers,
    current_frame,
    initial_playback_state,
    pause,
    play,
    queue_markers,
    restart,
    scrub,
    seconds_per_frame,
    set_speed,
    stop_for_stale_comparison,
)


@pytest.fixture(scope="module")
def official_replay():
    return build_comparative_replay(
        create_comparison_configuration(
            "staff_shortage",
            seed=404,
            duration_minutes=480,
            demand_multiplier=1.15,
            baseline_agents=3,
            assisted_agents=5,
            policy_mode="calibrated",
        )
    )


def test_supported_speeds_and_wall_clock_mapping_are_exact():
    assert SUPPORTED_SPEEDS == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert [seconds_per_frame(speed) for speed in SUPPORTED_SPEEDS] == [
        4.0,
        2.0,
        1.0,
        0.5,
        0.25,
    ]
    with pytest.raises(ValueError, match="unsupported replay speed"):
        seconds_per_frame(3)
    with pytest.raises(ValidationError):
        PlaybackState(speed_multiplier=3)


def test_play_pause_and_repeated_play_have_one_state_machine():
    initial = initial_playback_state(33)
    playing = play(initial, 33)

    assert initial.frame_index == 0
    assert playing.is_playing
    assert play(playing, 33) is playing
    assert pause(playing) == playing.model_copy(update={"is_playing": False})


def test_advance_stops_automatically_at_end_and_restart_preserves_speed():
    state = set_speed(play(initial_playback_state(3), 3), 8)
    state = advance(state, 3)
    assert state.frame_index == 1
    assert state.is_playing

    state = advance(state, 3)
    assert state.frame_index == 2
    assert not state.is_playing
    assert state.reached_end
    assert play(state, 3) is state

    restarted = restart(state, 3)
    assert restarted.frame_index == 0
    assert not restarted.is_playing
    assert not restarted.reached_end
    assert restarted.speed_multiplier == 8


def test_manual_scrub_pauses_and_speed_change_preserves_position():
    playing = play(initial_playback_state(33), 33)
    scrubbed = scrub(playing, 16, 33)

    assert scrubbed.frame_index == 16
    assert not scrubbed.is_playing
    faster = set_speed(scrubbed, 8)
    assert faster.frame_index == 16
    assert faster.speed_multiplier == 8
    assert not faster.is_playing


def test_stale_comparison_stops_playback_without_resetting_frame():
    state = scrub(initial_playback_state(33), 12, 33)
    state = play(state, 33)
    stopped = stop_for_stale_comparison(state)

    assert stopped.frame_index == 12
    assert not stopped.is_playing


def test_current_frame_keeps_both_sides_synchronized(official_replay):
    for index in range(len(official_replay.frames)):
        state = scrub(initial_playback_state(33), index, 33)
        frame = current_frame(official_replay, state)
        baseline_snapshot = official_replay.comparison.before.result.snapshots[index]
        assisted_snapshot = official_replay.comparison.after.result.snapshots[index]

        assert frame.index == index
        assert frame.simulation_minute == baseline_snapshot.simulation_time
        assert frame.simulation_minute == assisted_snapshot.simulation_time


def test_official_start_middle_end_display_values(official_replay):
    expected = {
        0: ((0, 0, 3, 0, 0, "LOW"), (0, 0, 5, 0, 0, "LOW")),
        16: ((0, 2, 1, 153, 47, "LOW"), (0, 2, 3, 192, 8, "LOW")),
        32: ((5, 3, 0, 289, 97, "HIGH"), (0, 4, 1, 373, 17, "LOW")),
    }

    for index, (baseline_expected, assisted_expected) in expected.items():
        frame = official_replay.frames[index]
        baseline = frame.baseline
        assisted = frame.assisted
        assert (
            baseline.queue_size,
            baseline.busy_agents,
            baseline.free_agents,
            baseline.completed_so_far,
            baseline.abandoned_so_far,
            baseline.snapshot_pressure.value,
        ) == baseline_expected
        assert (
            assisted.queue_size,
            assisted.busy_agents,
            assisted.free_agents,
            assisted.completed_so_far,
            assisted.abandoned_so_far,
            assisted.snapshot_pressure.value,
        ) == assisted_expected


def test_playback_operations_never_invoke_simulation(official_replay, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("playback must not invoke the Digital Twin")

    monkeypatch.setattr("callverse.dashboard.view_models.run_simulation", fail_if_called)
    state = initial_playback_state(len(official_replay.frames))
    state = play(state, len(official_replay.frames))
    state = advance(state, len(official_replay.frames))
    state = set_speed(state, 8)
    state = pause(state)
    state = scrub(state, 16, len(official_replay.frames))
    state = restart(state, len(official_replay.frames))
    assert current_frame(official_replay, state).index == 0


def test_queue_and_advisor_visuals_are_capped_without_losing_numeric_totals():
    queue = queue_markers(24, cap=10)
    advisors = advisor_markers(12, 5, cap=10)

    assert queue.markers == ("WAITING",) * 10
    assert queue.hidden_count == 14
    assert queue.total_count == 24
    assert advisors.markers == ("BUSY",) * 10
    assert advisors.hidden_count == 7
    assert advisors.total_count == 17


def test_streamlit_hides_controls_until_build_then_exposes_operational_view():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    assert not any(button.label == "PLAY" for button in app.button)

    next(
        button
        for button in app.button
        if button.label == "PREPARE PRIMARY TEACHING DEMO"
    ).click()
    app.run(timeout=30)
    next(button for button in app.button if button.label == "BUILD COMPARISON").click()
    app.run(timeout=30)

    assert not app.exception
    assert {"PLAY", "PAUSE", "RESTART"} <= {button.label for button in app.button}
    speed = next(box for box in app.selectbox if box.label == "Playback speed")
    assert tuple(speed.options) == ("0.5x", "1x", "2x", "4x", "8x")
    assert speed.value == 4
    metrics = {metric.label for metric in app.metric}
    assert "SIMULATED TIME" in metrics
    assert {"Current queue", "Busy advisors", "Free advisors"} <= metrics
    assert {"Completed so far", "Abandoned so far"} <= metrics

    timeline = next(
        slider
        for slider in app.select_slider
        if slider.label == "Synchronized simulated time"
    )
    timeline.set_value(16)
    app.run(timeout=30)
    assert next(
        metric.value for metric in app.metric if metric.label == "SIMULATED TIME"
    ) == "12:00"

    speed = next(box for box in app.selectbox if box.label == "Playback speed")
    speed.set_value(8.0)
    app.run(timeout=30)
    assert next(
        metric.value for metric in app.metric if metric.label == "SIMULATED TIME"
    ) == "12:00"

    next(button for button in app.button if button.label == "RESTART").click()
    app.run(timeout=30)
    assert next(
        metric.value for metric in app.metric if metric.label == "SIMULATED TIME"
    ) == "08:00"
    assert next(
        box.value for box in app.selectbox if box.label == "Playback speed"
    ) == 8.0


def test_streamlit_play_reaches_end_without_rerunning_simulation(monkeypatch):
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    next(
        button
        for button in app.button
        if button.label == "PREPARE PRIMARY TEACHING DEMO"
    ).click()
    app.run(timeout=30)
    next(button for button in app.button if button.label == "BUILD COMPARISON").click()
    app.run(timeout=30)
    next(box for box in app.selectbox if box.label == "Playback speed").set_value(8.0)
    app.run(timeout=30)

    monkeypatch.setattr("callverse.dashboard.manager.sleep", lambda _seconds: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("UI playback must not invoke the Digital Twin")

    monkeypatch.setattr("callverse.dashboard.view_models.run_simulation", fail_if_called)
    next(button for button in app.button if button.label == "PLAY").click()
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Replay complete" for item in app.success)
    assert next(
        metric.value for metric in app.metric if metric.label == "SIMULATED TIME"
    ) == "16:00"
