from __future__ import annotations

from time import perf_counter

import pytest
from streamlit.testing.v1 import AppTest

from callverse.dashboard.twin_replay import (
    ReplayPressure,
    build_replay_frame,
    replay_context,
    replay_option_label,
    replay_slider_key,
    waiting_visual,
)
from callverse.dashboard.view_models import ManagerRun, run_manager_simulation
from callverse.scenarios import get_scenario


@pytest.fixture(scope="module")
def replay_runs() -> dict[str, ManagerRun]:
    return {
        name: run_manager_simulation(get_scenario(name), "calibrated")
        for name in ("normal_day", "staff_shortage", "perfect_storm")
    }


def test_replay_selects_existing_snapshot_without_running_simulator(
    replay_runs, monkeypatch
):
    run = replay_runs["staff_shortage"]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("replay must not invoke the simulator")

    monkeypatch.setattr("callverse.simulation.run_simulation", fail_if_called)
    middle = len(run.result.snapshots) // 2
    frame = build_replay_frame(run, middle)
    snapshot = run.result.snapshots[middle]

    assert frame.simulation_minute == snapshot.simulation_time
    assert frame.queue_size == snapshot.queue_size
    assert frame.completed_count == snapshot.completed_count
    assert frame.abandoned_count == snapshot.abandoned_count


def test_replay_frame_does_not_mutate_simulation_result(replay_runs):
    run = replay_runs["normal_day"]
    before = run.model_dump()
    build_replay_frame(run, len(run.result.snapshots) // 2)
    assert run.model_dump() == before


def test_agent_busy_and_free_counts_are_exact(replay_runs):
    run = replay_runs["perfect_storm"]
    frame = build_replay_frame(run, len(run.result.snapshots) // 2)
    assert frame.busy_agents + frame.free_agents == frame.available_agents
    assert frame.available_agents == run.result.available_agents == 4


def test_large_queue_visual_is_capped(replay_runs):
    run = replay_runs["perfect_storm"]
    snapshot = run.result.snapshots[0].model_copy(
        update={"queue_size": 300, "busy_agents": run.result.available_agents}
    )
    result = run.result.model_copy(update={"snapshots": (snapshot,)})
    synthetic_view = run.model_copy(update={"result": result})

    frame = build_replay_frame(synthetic_view, 0, icon_cap=10)
    visual = waiting_visual(frame)
    assert frame.visible_waiting_icons == 10
    assert frame.hidden_waiting_count == 290
    assert visual.count("👤") == 10
    assert "+290 more" in visual


def test_snapshot_counts_are_labelled_and_remain_cumulative(replay_runs):
    for run in replay_runs.values():
        completed = [snapshot.completed_count for snapshot in run.result.snapshots]
        abandoned = [snapshot.abandoned_count for snapshot in run.result.snapshots]
        assert completed == sorted(completed)
        assert abandoned == sorted(abandoned)


def test_scenario_context_matches_latest_run(replay_runs):
    context = replay_context(replay_runs["staff_shortage"])
    assert context.scenario_title == "Staff Shortage"
    assert context.risk_level == "High"
    assert context.available_agents == 3
    assert context.seed == 404
    assert context.policy_mode == "calibrated"


def test_replay_state_key_changes_with_scenario_configuration(replay_runs):
    normal = replay_runs["normal_day"]
    shortage = replay_runs["staff_shortage"]
    assert replay_slider_key(normal) != replay_slider_key(shortage)

    changed_agents = shortage.scenario.model_copy(update={"available_agents": 5})
    changed_run = shortage.model_copy(update={"scenario": changed_agents})
    assert replay_slider_key(shortage) != replay_slider_key(changed_run)


def test_human_readable_replay_time_uses_simulated_clock(replay_runs):
    run = replay_runs["normal_day"]
    assert replay_option_label(run, 0) == "08:00 · minute 0"
    middle = len(run.result.snapshots) // 2
    assert ":" in replay_option_label(run, middle)


def test_pressure_categories_are_transparent(replay_runs):
    run = replay_runs["staff_shortage"]
    low = run.result.snapshots[0].model_copy(update={"queue_size": 0, "busy_agents": 0})
    high = low.model_copy(update={"queue_size": 3, "busy_agents": 3})
    severe = low.model_copy(update={"queue_size": 6, "busy_agents": 3})
    result = run.result.model_copy(update={"snapshots": (low, high, severe)})
    view = run.model_copy(update={"result": result})

    assert build_replay_frame(view, 0).pressure is ReplayPressure.LOW
    assert build_replay_frame(view, 1).pressure is ReplayPressure.HIGH
    assert build_replay_frame(view, 2).pressure is ReplayPressure.SEVERE


def test_three_scenarios_show_distinct_actual_pressure_patterns(replay_runs):
    normal = replay_runs["normal_day"].result
    shortage = replay_runs["staff_shortage"].result
    storm = replay_runs["perfect_storm"].result

    assert max(snapshot.queue_size for snapshot in normal.snapshots) < max(
        snapshot.queue_size for snapshot in shortage.snapshots
    )
    assert max(snapshot.queue_size for snapshot in shortage.snapshots) < max(
        snapshot.queue_size for snapshot in storm.snapshots
    )
    assert normal.counts.abandoned < shortage.counts.abandoned < storm.counts.abandoned


def test_replay_frame_construction_is_lightweight(replay_runs):
    run = replay_runs["perfect_storm"]
    started = perf_counter()
    for index in range(len(run.result.snapshots)):
        build_replay_frame(run, index)
    assert perf_counter() - started < 0.1


def test_default_manager_empty_replay_and_supporting_navigation_render():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "📊 Manager Control Room"
    assert any(item.value == "CallVerse · Manager Control Room" for item in app.title)
    assert any("unlock the replay" in item.value for item in app.info)

    app.sidebar.radio[0].set_value("💬 Customer Interaction Demo")
    app.run(timeout=30)
    assert not app.exception
    customer_copy = "\n".join(item.value for item in app.caption)
    assert "Demonstrates the Advisor pipeline" in customer_copy

    app.sidebar.radio[0].set_value("🧑‍💼 Human Approval Queue")
    app.run(timeout=30)
    assert not app.exception
    staff_copy = "\n".join(item.value for item in app.caption)
    assert "human-in-the-loop review" in staff_copy


def test_completed_run_unlocks_replay_without_replacing_existing_charts():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(button for button in app.button if button.label == "RUN DIGITAL TWIN").click()
    app.run(timeout=30)

    assert not app.exception
    subheaders = {item.value for item in app.subheader}
    assert "Digital Twin Replay" in subheaders
    assert "Full-run monitoring" in subheaders
    metrics = {item.label: item.value for item in app.metric}
    assert "Simulated time" in metrics
    assert "Waiting contacts" in metrics
    assert "Agents busy / free" in metrics
    assert "Cumulative completed" in metrics
    visible = "\n".join(item.value for item in app.caption)
    assert "does not run a second simulation" in visible
