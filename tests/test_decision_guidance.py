from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from callverse.dashboard.decision_guidance import (
    PRODUCTION_AB_DISCLAIMER,
    SAME_SEED_EXPLANATION,
    ChangeAssessment,
    DecisionOutcome,
    build_decision_narrative,
    format_metric_delta,
    metric_changes,
    recommended_decision_widget_state,
)
from callverse.dashboard.view_models import (
    DecisionComparison,
    ManagerRun,
    run_manager_simulation,
    run_staffing_what_if,
)
from callverse.scenarios import get_scenario


@pytest.fixture(scope="module")
def staff_comparisons() -> dict[int, DecisionComparison]:
    before = run_manager_simulation(get_scenario("staff_shortage"), "calibrated")
    return {agents: run_staffing_what_if(before, agents) for agents in (2, 3, 4, 5)}


def _with_after_metrics(
    comparison: DecisionComparison,
    *,
    sla: float,
    abandonment: float,
    wait: float,
    occupancy: float,
) -> DecisionComparison:
    after_result = comparison.after.result
    changed_result = after_result.model_copy(
        update={
            "kpis": after_result.kpis.model_copy(
                update={
                    "sla": sla,
                    "abandonment_rate": abandonment,
                    "average_waiting_time": wait,
                    "occupancy": occupancy,
                }
            )
        }
    )
    return comparison.model_copy(
        update={
            "after": ManagerRun(
                scenario=comparison.after.scenario,
                policy_mode=comparison.after.policy_mode,
                result=changed_result,
            )
        }
    )


def test_official_delta_units_and_directions(staff_comparisons):
    changes = {change.key: change for change in metric_changes(staff_comparisons[5])}

    assert format_metric_delta(changes["sla"]).endswith("percentage points")
    assert format_metric_delta(changes["abandonment"]).endswith("percentage points")
    assert format_metric_delta(changes["occupancy"]).endswith("percentage points")
    assert format_metric_delta(changes["average_wait"]).endswith("min")
    assert format_metric_delta(changes["completed"]).endswith("contacts")
    assert format_metric_delta(changes["final_backlog"]).endswith("contacts")
    assert changes["final_backlog"].label == "Final queue backlog"
    assert changes["sla"].assessment is ChangeAssessment.BETTER
    assert changes["abandonment"].assessment is ChangeAssessment.BETTER
    assert changes["average_wait"].assessment is ChangeAssessment.BETTER


def test_occupancy_direction_is_target_aware(staff_comparisons):
    comparison = staff_comparisons[5]
    before = _with_after_metrics(
        comparison,
        sla=comparison.before.result.kpis.sla,
        abandonment=comparison.before.result.kpis.abandonment_rate,
        wait=comparison.before.result.kpis.average_waiting_time,
        occupancy=0.60,
    )
    before_result = before.before.result.model_copy(
        update={
            "kpis": before.before.result.kpis.model_copy(update={"occupancy": 0.70})
        }
    )
    comparison = before.model_copy(
        update={
            "before": ManagerRun(
                scenario=before.before.scenario,
                policy_mode=before.before.policy_mode,
                result=before_result,
            )
        }
    )
    occupancy = next(
        change for change in metric_changes(comparison) if change.key == "occupancy"
    )
    assert occupancy.assessment is ChangeAssessment.NEUTRAL
    assert occupancy.preference == "target"


def test_official_target_checks_move_from_failures_to_passes(staff_comparisons):
    targets = {
        target.metric: target
        for target in build_decision_narrative(staff_comparisons[5]).targets
    }
    assert (targets["SLA"].before, targets["SLA"].after) == ("fail", "pass")
    assert (targets["Abandonment"].before, targets["Abandonment"].after) == (
        "fail",
        "pass",
    )
    assert (targets["Occupancy"].before, targets["Occupancy"].after) == (
        "warning",
        "pass",
    )


def test_strongly_improved_outcome_and_staffing_tradeoff(staff_comparisons):
    narrative = build_decision_narrative(staff_comparisons[5])
    assert narrative.outcome is DecisionOutcome.STRONGLY_IMPROVED
    assert narrative.action == "Increase staffing from 3 to 5 agents."
    assert "2 additional agents" in narrative.trade_off
    assert "Forecast and Workforce" in narrative.next_step


def test_improved_outcome(staff_comparisons):
    narrative = build_decision_narrative(staff_comparisons[4])
    assert narrative.outcome is DecisionOutcome.IMPROVED


def test_mixed_outcome_when_service_improves_but_center_remains_overloaded(
    staff_comparisons,
):
    comparison = _with_after_metrics(
        staff_comparisons[4],
        sla=0.70,
        abandonment=0.20,
        wait=1.50,
        occupancy=0.82,
    )
    narrative = build_decision_narrative(comparison)
    assert narrative.outcome is DecisionOutcome.MIXED_TRADE_OFF
    assert "remain unresolved" in narrative.conclusion


def test_little_change_outcome(staff_comparisons):
    narrative = build_decision_narrative(staff_comparisons[3])
    assert narrative.outcome is DecisionOutcome.LITTLE_CHANGE
    assert "No staffing trade-off" in narrative.trade_off


def test_worsened_outcome(staff_comparisons):
    narrative = build_decision_narrative(staff_comparisons[2])
    assert narrative.outcome is DecisionOutcome.WORSENED
    assert narrative.action == "Reduce staffing from 3 to 2 agents."


def test_same_seed_and_production_disclaimer_are_explicit():
    assert "same random seed" in SAME_SEED_EXPLANATION
    assert "reduces random variation" in SAME_SEED_EXPLANATION
    assert "not a production A/B test" in PRODUCTION_AB_DISCLAIMER


def test_recommended_decision_prefill_has_no_run_state():
    state = recommended_decision_widget_state("manager_after_agents_demo")
    assert state == {"manager_after_agents_demo": 5}
    assert "manager_comparison" not in state


def test_narrative_preserves_same_seed_context_and_does_not_mutate_comparison(
    staff_comparisons,
):
    comparison = staff_comparisons[5]
    before = comparison.model_dump()
    narrative = build_decision_narrative(comparison)

    assert comparison.model_dump() == before
    assert comparison.before.result.seed == comparison.after.result.seed == 404
    assert narrative.changed_parameter == "Available agents: 3 → 5"
    held = " ".join(narrative.held_constant)
    assert "Seed: 404" in held
    assert "Simulator mode: calibrated" in held
    assert "Demand multiplier: 1.15x" in held


def test_dashboard_renders_full_decision_story_without_auto_run():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)
    next(
        button
        for button in app.button
        if button.label == "LOAD JURY STARTING SCENARIO"
    ).click()
    app.run(timeout=30)
    next(button for button in app.button if button.label == "RUN DIGITAL TWIN").click()
    app.run(timeout=30)

    assert not app.exception
    visible = "\n".join(
        item.value for item in (*app.info, *app.caption, *app.subheader)
    )
    assert "Inherited from Scenario Studio" in visible
    assert "same random seed" in visible
    assert "not a production A/B test" in visible

    next(
        button
        for button in app.button
        if button.label == "PREPARE CAPACITY WHAT-IF DECISION"
    ).click()
    app.run(timeout=30)
    after_input = next(
        widget
        for widget in app.number_input
        if widget.label == "AFTER available agents"
    )
    assert after_input.value == 5
    assert not any("MANAGER CONCLUSION" in item.value for item in app.subheader)

    next(button for button in app.button if button.label == "RUN COMPARISON").click()
    app.run(timeout=30)
    assert not app.exception
    headings = "\n".join(item.value for item in app.subheader)
    assert "1 · PROBLEM" in headings
    assert "2 · PROPOSED ACTION" in headings
    assert "3 · SIMULATED EFFECT" in headings
    assert "4 · MANAGER CONCLUSION" in headings
    conclusion = "\n".join(item.value for item in app.success)
    assert "STRONGLY IMPROVED" in conclusion
    assert any(metric.label == "SLA" for metric in app.metric)
