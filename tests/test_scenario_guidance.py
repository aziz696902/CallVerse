from __future__ import annotations

from streamlit.testing.v1 import AppTest

from callverse.dashboard.scenario_guidance import (
    DEFAULT_TARGETS,
    RECOMMENDED_DEMO,
    SCENARIO_GUIDES,
    CenterStatus,
    get_scenario_guide,
    interpret_simulation_result,
    recommended_demo_widget_state,
)
from callverse.dashboard.view_models import policy_for, run_manager_simulation
from callverse.scenarios import SCENARIO_PRESETS, get_scenario
from callverse.simulation import run_simulation


def _result_with_metrics(*, sla: float, abandonment: float, occupancy: float):
    result = run_manager_simulation(get_scenario("normal_day"), "calibrated").result
    return result.model_copy(
        update={
            "kpis": result.kpis.model_copy(
                update={
                    "sla": sla,
                    "abandonment_rate": abandonment,
                    "occupancy": occupancy,
                }
            )
        }
    )


def test_every_scenario_has_exactly_one_guidance_record():
    scenario_keys = {scenario.name for scenario in SCENARIO_PRESETS}
    assert set(SCENARIO_GUIDES) == scenario_keys
    assert len(SCENARIO_GUIDES) == len(SCENARIO_PRESETS) == 7
    assert all(guide.scenario_key == key for key, guide in SCENARIO_GUIDES.items())


def test_recommended_demo_and_loader_are_explicit_and_do_not_run_a_simulation():
    state = recommended_demo_widget_state()
    assert RECOMMENDED_DEMO.scenario_key == "staff_shortage"
    assert RECOMMENDED_DEMO.seed == 404
    assert RECOMMENDED_DEMO.available_agents == 3
    assert RECOMMENDED_DEMO.policy_mode == "calibrated"
    assert state == {
        "manager_preset": "staff_shortage",
        "manager_seed_staff_shortage": 404,
        "manager_agents_staff_shortage": 3,
        "manager_policy_mode": "calibrated",
    }
    assert "manager_run" not in state


def test_guidance_has_no_hardcoded_verified_simulation_outputs():
    prose = " ".join(
        value
        for guide in SCENARIO_GUIDES.values()
        for value in (
            guide.short_description,
            guide.situation,
            guide.manager_question,
            guide.expected_behavior,
            guide.suggested_action,
            guide.scientific_note,
        )
    )
    for forbidden in ("352", "394", "289", "934", "64.7%", "24.6%", "85.0%", "27.9%"):
        assert forbidden not in prose
    assert "guaranteed" not in prose.lower()
    assert "proves" not in prose.lower()


def test_reading_guidance_does_not_mutate_frozen_scenario_config():
    before = [scenario.model_dump() for scenario in SCENARIO_PRESETS]
    for scenario in SCENARIO_PRESETS:
        get_scenario_guide(scenario.name)
    after = [scenario.model_dump() for scenario in SCENARIO_PRESETS]
    assert after == before


def test_healthy_interpretation_and_all_target_passes():
    interpretation = interpret_simulation_result(
        _result_with_metrics(sla=0.90, abandonment=0.05, occupancy=0.70)
    )
    assert interpretation.status is CenterStatus.HEALTHY
    assert [check.outcome for check in interpretation.target_checks] == [
        "pass",
        "pass",
        "pass",
    ]
    assert interpretation.next_step.startswith("Open Forecast")


def test_overloaded_interpretation_and_target_failures():
    interpretation = interpret_simulation_result(
        _result_with_metrics(sla=0.65, abandonment=0.20, occupancy=0.86)
    )
    assert interpretation.status is CenterStatus.OVERLOADED
    assert [check.outcome for check in interpretation.target_checks] == [
        "fail",
        "fail",
        "warning",
    ]
    assert "Compare Decisions" in interpretation.next_step


def test_critical_interpretation_uses_transparent_severe_rule():
    interpretation = interpret_simulation_result(
        _result_with_metrics(sla=0.39, abandonment=0.41, occupancy=0.97)
    )
    assert interpretation.status is CenterStatus.CRITICAL
    assert "Compare Decisions" in interpretation.next_step


def test_target_boundaries_match_managerial_v1_assumptions():
    interpretation = interpret_simulation_result(
        _result_with_metrics(
            sla=DEFAULT_TARGETS.minimum_sla,
            abandonment=DEFAULT_TARGETS.maximum_abandonment,
            occupancy=DEFAULT_TARGETS.maximum_occupancy,
        )
    )
    checks = {check.metric: check.outcome for check in interpretation.target_checks}
    assert checks == {"SLA": "pass", "Abandonment": "fail", "Occupancy": "warning"}


def test_guidance_layer_does_not_change_simulation_result():
    scenario = get_scenario("staff_shortage")
    direct = run_simulation(
        scenario, seed=scenario.random_seed, policy=policy_for("calibrated")
    )
    guided = run_manager_simulation(scenario, "calibrated").result
    assert guided == direct


def test_dashboard_renders_guided_journey_and_scenario_card():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)

    assert not app.exception
    visible = "\n".join(
        item.value for item in (*app.info, *app.caption, *app.markdown, *app.subheader)
    )
    assert "1 Scenario" in visible
    assert "Manager question" in visible
    assert "Operational objectives" in visible
    next(
        button for button in app.button if button.label == "LOAD JURY STARTING SCENARIO"
    ).click()
    app.run(timeout=30)

    assert not app.exception
    selectboxes = {widget.label: widget.value for widget in app.selectbox}
    number_inputs = {widget.label: widget.value for widget in app.number_input}
    assert selectboxes["Scenario preset"] == "staff_shortage"
    assert selectboxes["Policy mode"] == "calibrated"
    assert number_inputs["Simulation seed"] == 404
    assert number_inputs["Available agents"] == 3
    assert not any(item.value == "Latest operational result" for item in app.subheader)
