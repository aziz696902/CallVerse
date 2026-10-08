from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from callverse.customer_advisor import (
    AdvisorInteraction,
    InteractionMetadata,
    RoutingPath,
)
from callverse.dashboard.manager import SCENARIO_NAMES
from callverse.dashboard.view_models import (
    comparison_rows,
    configure_scenario,
    export_payload,
    format_minutes,
    format_percentage,
    interaction_summary,
    kpi_cards,
    policy_for,
    run_manager_simulation,
    run_staffing_what_if,
    scenario_warnings,
    session_quality_summary,
    timeline_rows,
)
from callverse.domain import AdvisorResult, RequestIntent
from callverse.scenarios import SCENARIO_PRESETS
from callverse.simulation import DEFAULT_POLICY, SimulationResult


def short_scenario(name: str = "normal_day", *, agents: int = 4):
    return configure_scenario(
        name,
        seed=818,
        available_agents=agents,
        demand_multiplier=1.1,
        duration=120,
    )


def test_all_seven_scenarios_are_represented():
    assert SCENARIO_NAMES == tuple(scenario.name for scenario in SCENARIO_PRESETS)
    assert SCENARIO_NAMES == (
        "normal_day",
        "rainy_peak",
        "flash_sale",
        "staff_shortage",
        "customer_crisis",
        "knowledge_failure",
        "perfect_storm",
    )


def test_scenario_controls_change_only_active_fields():
    preset = next(item for item in SCENARIO_PRESETS if item.name == "rainy_peak")
    configured = configure_scenario(
        "rainy_peak", seed=99, available_agents=11, demand_multiplier=1.7, duration=300
    )

    assert configured.random_seed == 99
    assert configured.available_agents == 11
    assert configured.demand_multiplier == 1.7
    assert configured.simulation_duration == 300
    assert configured.external_condition == preset.external_condition
    assert configured.late_delivery_rate == preset.late_delivery_rate
    assert configured.knowledge_base_state == preset.knowledge_base_state
    assert configured.ai_advisor_enabled == preset.ai_advisor_enabled


def test_prototype_dashboard_run_uses_real_simulation_result():
    run = run_manager_simulation(short_scenario(), "prototype")

    assert isinstance(run.result, SimulationResult)
    assert run.result.counts.generated > 0
    assert run.policy_mode == "prototype"
    assert policy_for("prototype") is DEFAULT_POLICY


def test_calibrated_dashboard_run_works_offline():
    run = run_manager_simulation(short_scenario(), "calibrated")

    assert isinstance(run.result, SimulationResult)
    assert run.result.counts.generated > 0
    assert run.policy_mode == "calibrated"
    assert policy_for("calibrated").arrival_slot_multipliers is not None


def test_kpi_formatting_preserves_unavailable_values():
    assert format_minutes(None) == "N/A"
    assert format_percentage(None) == "N/A"

    run = run_manager_simulation(short_scenario(), "prototype")
    cards = {card.label: card.value for card in kpi_cards(run.result)}
    assert len(cards) == 7
    assert cards["Generated contacts"] != "N/A"
    assert cards["SLA"].endswith("%")


def test_time_series_conversion_matches_real_snapshots():
    result = run_manager_simulation(short_scenario(), "prototype").result
    rows = timeline_rows(result)

    assert len(rows) == len(result.snapshots)
    assert rows[0]["simulation_time"] == result.snapshots[0].simulation_time
    assert rows[-1]["queue_size"] == result.snapshots[-1].queue_size
    assert rows[-1]["completed"] == result.snapshots[-1].completed_count


def test_staffing_what_if_uses_same_seed_and_changes_only_agents():
    before = run_manager_simulation(short_scenario("staff_shortage", agents=3), "calibrated")
    comparison = run_staffing_what_if(before, 5)
    before_config = before.scenario.model_dump()
    after_config = comparison.after.scenario.model_dump()

    assert comparison.before.result.seed == comparison.after.result.seed == 818
    assert comparison.before.policy_mode == comparison.after.policy_mode == "calibrated"
    assert comparison.after.scenario.available_agents == 5
    before_config.pop("available_agents")
    after_config.pop("available_agents")
    assert before_config == after_config
    assert comparison.before.result.counts.generated == comparison.after.result.counts.generated


def test_comparison_delta_calculation_is_exact():
    before = run_manager_simulation(short_scenario("staff_shortage", agents=3), "prototype")
    comparison = run_staffing_what_if(before, 5)
    rows = {row.metric: row for row in comparison_rows(comparison)}

    expected_sla = round(
        (comparison.after.result.kpis.sla - comparison.before.result.kpis.sla) * 100, 3
    )
    expected_wait = round(
        comparison.after.result.kpis.average_waiting_time
        - comparison.before.result.kpis.average_waiting_time,
        3,
    )
    assert rows["SLA"].delta == expected_sla
    assert rows["Average wait"].delta == expected_wait


def test_warning_thresholds_are_descriptive_and_explicit():
    run = run_manager_simulation(
        configure_scenario(
            "perfect_storm",
            seed=12,
            available_agents=2,
            demand_multiplier=2.5,
            duration=120,
        ),
        "prototype",
    )
    warnings = scenario_warnings(run.result)

    assert warnings
    assert any("SLA" in warning or "abandonment" in warning for warning in warnings)


def test_compact_export_contains_configuration_kpis_and_comparison_only():
    before = run_manager_simulation(short_scenario("staff_shortage", agents=3), "prototype")
    comparison = run_staffing_what_if(before, 5)
    payload = export_payload(before, comparison)
    encoded = json.dumps(payload)

    assert payload["configuration"]["seed"] == 818
    assert payload["comparison"]["same_seed"] is True
    assert "kpis" in payload
    assert "request_records" not in encoded
    assert "customer_message" not in encoded


def test_interaction_metadata_is_rendered_as_compact_observable_values():
    interaction = AdvisorInteraction(
        result=AdvisorResult(
            request_id="LAB-1",
            resolved=True,
            escalated=False,
            automated=True,
            response_text="Verified response",
        ),
        metadata=InteractionMetadata(
            routing_path=RoutingPath.CLASSIFIER,
            predicted_intent=RequestIntent.TRACKING,
            classifier_confidence=0.9,
            classifier_accepted=True,
            extracted_order_id="ORD-5003",
            tools_used=("get_order", "get_tracking"),
            resolved=True,
            escalated=False,
        ),
    )

    summary = interaction_summary(interaction)

    assert summary["predicted_intent"] == "tracking"
    assert summary["tools_used"] == ["get_order", "get_tracking"]
    assert "response_text" not in summary


def test_empty_quality_session_does_not_manufacture_aggregate():
    assert session_quality_summary([]) is None


def test_streamlit_customer_staff_and_manager_views_load_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "📊 Manager Control Room"
    assert any(item.value == "CallVerse · Manager Control Room" for item in app.title)

    app.sidebar.radio[0].set_value("💬 Customer Interaction Demo")
    app.run()
    assert not app.exception
    assert any(item.value == "💬 Customer Interaction Demo" for item in app.title)

    app.sidebar.radio[0].set_value("🧑‍💼 Human Approval Queue")
    app.run()
    assert not app.exception
    assert any(item.value == "🧑‍💼 Human Approval Queue" for item in app.title)

    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run()
    assert not app.exception
    assert [item.label for item in app.tabs] == [
        "Scenario Studio",
        "Twin Monitor",
        "Forecast",
        "Workforce",
        "Compare Decisions",
        "Interaction Lab",
        "Quality",
    ]


def test_streamlit_manager_run_button_returns_real_kpis():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run()
    next(button for button in app.button if button.label == "RUN DIGITAL TWIN").click()
    app.run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Generated contacts" in metrics
    assert int(metrics["Generated contacts"].replace(",", "")) > 0
    assert metrics["Average wait"].endswith(" min")


def test_streamlit_forecast_view_is_reachable_and_honestly_labelled():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)

    assert not app.exception
    assert any(item.value == "Demand Forecast" for item in app.header)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert float(metrics["Predicted next-24h contacts"]) > 0
    visible_text = "\n".join(item.value for item in (*app.caption, *app.warning))
    assert "ACTUAL HISTORY" in visible_text
    assert "FORECAST covers" in visible_text
    assert "does not recommend staffing levels" in visible_text


def test_streamlit_workforce_view_builds_an_honestly_labelled_plan():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)
    assert not app.exception
    assert any(item.value == "Workforce" for item in app.header)
    next(button for button in app.button if button.label == "BUILD WORKFORCE PLAN").click()
    app.run(timeout=30)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["Peak planned agents"]) > 0
    assert float(metrics["Total agent-hours"]) > 0
    visible_text = "\n".join(item.value for item in (*app.caption, *app.warning, *app.info))
    assert "Erlang-C analytical staffing recommendation" in visible_text
    assert "Theoretical Erlang-C predictions" in visible_text
    assert "Dynamic 30-minute staffing" in visible_text


def test_streamlit_workforce_view_shows_experimental_ppo_limitations():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)

    assert not app.exception
    visible_text = "\n".join(
        item.value for item in (*app.subheader, *app.caption, *app.error, *app.warning)
    )
    assert "EXPERIMENTAL PPO POLICY — NOT ADOPTED" in visible_text
    assert "Operational recommendation: NOT ADOPTED" in visible_text
    assert "simplified training environment" in visible_text


def test_streamlit_offline_interaction_is_explicit_and_quality_scores_stay_unavailable():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run()
    next(button for button in app.button if button.label == "RUN SELECTED INTERACTION").click()
    app.run(timeout=30)

    assert not app.exception
    visible_text = "\n".join(item.value for item in (*app.info, *app.markdown, *app.caption))
    assert "Offline deterministic demo mode" in visible_text
    assert "Evaluation status: **unavailable**" in visible_text
    assert "No six-dimension or overall scores" in visible_text
    assert not any(metric.label == "Overall quality" for metric in app.metric)
