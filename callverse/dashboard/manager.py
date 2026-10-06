"""Streamlit composition for the CallVerse Manager Control Room."""

from __future__ import annotations

import uuid

import streamlit as st

from callverse.customer_advisor import CallVerseCustomerAdvisor
from callverse.customer_advisor_demo import deterministic_demo_runner
from callverse.domain import RequestIntent, SupportRequest
from callverse.forecasting.service import (
    forecast_chart_rows,
    forecast_next_24h,
    forecast_summary,
)
from callverse.quality import QualityAnalyst, QualityEvaluationInput
from callverse.quality.benchmark import load_policy_quality_benchmark
from callverse.scenarios import SCENARIO_PRESETS, get_scenario
from helppilot import config

from .view_models import (
    DecisionComparison,
    ManagerRun,
    comparison_table,
    configure_scenario,
    export_json,
    intent_mix_rows,
    interaction_summary,
    kpi_cards,
    persona_mix_rows,
    run_manager_simulation,
    run_staffing_what_if,
    scenario_warnings,
    session_quality_summary,
    timeline_rows,
)

SCENARIO_NAMES = tuple(scenario.name for scenario in SCENARIO_PRESETS)
PROJECT_ROOT = config.PROJECT_ROOT
FORECAST_SERIES_PATH = PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv"
FORECAST_ARTIFACT_PATH = PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"

DEMO_MESSAGES = {
    "Grounded tracking": ("CUST-1003", "Track my order ORD-5003"),
    "Refund approval": ("CUST-1001", "I want a refund for ORD-5001"),
    "Damaged item fallback": ("CUST-1004", "My item arrived broken"),
    "Low-confidence fallback": ("CUST-1003", "Where is my order?"),
    "Missing order": ("CUST-1003", "Track ORD-9999"),
}


def _render_kpis(run: ManagerRun) -> None:
    cards = kpi_cards(run.result)
    first = st.columns(4)
    second = st.columns(3)
    for column, card in zip((*first, *second), cards):
        column.metric(card.label, card.value)


def _render_reproducibility(run: ManagerRun) -> None:
    with st.expander("Run configuration and reproducibility"):
        st.json(
            {
                "scenario": run.scenario.name,
                "seed": run.result.seed,
                "policy_mode": run.policy_mode,
                "available_agents": run.scenario.available_agents,
                "demand_multiplier": run.scenario.demand_multiplier,
                "simulation_duration_minutes": run.scenario.simulation_duration,
            }
        )


def _render_run_summary(run: ManagerRun) -> None:
    st.subheader("Latest operational result")
    _render_kpis(run)
    warnings = scenario_warnings(run.result)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No V1 operational warning threshold was triggered.")
    _render_reproducibility(run)


def _scenario_studio() -> None:
    st.header("Scenario Studio")
    st.write(
        "Run a virtual support-center scenario, observe the operational impact, "
        "then test a decision before applying it."
    )

    preset_name = st.selectbox("Scenario preset", SCENARIO_NAMES, key="manager_preset")
    preset = get_scenario(preset_name)
    st.caption(preset.description or "")
    c1, c2, c3 = st.columns(3)
    seed = int(
        c1.number_input(
            "Simulation seed",
            min_value=0,
            value=preset.random_seed,
            step=1,
            key=f"manager_seed_{preset_name}",
        )
    )
    agents = int(
        c2.number_input(
            "Available agents",
            min_value=1,
            value=preset.available_agents,
            step=1,
            key=f"manager_agents_{preset_name}",
        )
    )
    policy_mode = c3.selectbox("Policy mode", ("calibrated", "prototype"))
    c4, c5 = st.columns(2)
    demand = float(
        c4.number_input(
            "Demand multiplier",
            min_value=0.1,
            max_value=5.0,
            value=float(preset.demand_multiplier),
            step=0.05,
            key=f"manager_demand_{preset_name}",
        )
    )
    duration = float(
        c5.number_input(
            "Duration (simulated minutes)",
            min_value=30.0,
            max_value=1440.0,
            value=float(preset.simulation_duration),
            step=30.0,
            key=f"manager_duration_{preset_name}",
        )
    )

    with st.expander("Scenario context — not yet causally connected"):
        st.caption(
            "These fields describe the scenario but do not currently alter queue mechanics."
        )
        st.json(
            {
                "external_condition": preset.external_condition,
                "late_delivery_rate": preset.late_delivery_rate,
                "knowledge_base_state": preset.knowledge_base_state.value,
                "ai_advisor_enabled": preset.ai_advisor_enabled,
            }
        )

    if st.button("RUN DIGITAL TWIN", type="primary", use_container_width=True):
        scenario = configure_scenario(
            preset_name,
            seed=seed,
            available_agents=agents,
            demand_multiplier=demand,
            duration=duration,
        )
        with st.spinner("Running the CallVerse Digital Twin…"):
            st.session_state.manager_run = run_manager_simulation(scenario, policy_mode)
        st.session_state.manager_comparison = None

    run: ManagerRun | None = st.session_state.get("manager_run")
    if run is None:
        st.info("Choose assumptions and press RUN DIGITAL TWIN. The model does not auto-run.")
        return
    _render_run_summary(run)
    comparison: DecisionComparison | None = st.session_state.get("manager_comparison")
    st.download_button(
        "Download compact JSON result",
        data=export_json(run, comparison),
        file_name=f"callverse-{run.scenario.name}-{run.result.seed}.json",
        mime="application/json",
    )


def _twin_monitor() -> None:
    st.header("Twin Monitor")
    run: ManagerRun | None = st.session_state.get("manager_run")
    if run is None:
        st.info("Run a scenario in Scenario Studio to populate operational monitoring.")
        return
    rows = timeline_rows(run.result)
    st.subheader("Queue size over simulated time")
    st.line_chart(rows, x="simulation_time", y="queue_size", x_label="Simulated minutes")
    st.subheader("Busy agents over simulated time")
    st.line_chart(rows, x="simulation_time", y="busy_agents", x_label="Simulated minutes")
    st.subheader("Cumulative completed and abandoned contacts")
    st.line_chart(
        rows,
        x="simulation_time",
        y=["completed", "abandoned"],
        x_label="Simulated minutes",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contacts by intent")
        st.bar_chart(intent_mix_rows(run.result), x="intent", y="contacts")
    with c2:
        st.subheader("Contacts by persona")
        st.bar_chart(persona_mix_rows(run.result), x="persona", y="contacts")


def _compare_decisions() -> None:
    st.header("Compare Decisions")
    run: ManagerRun | None = st.session_state.get("manager_run")
    if run is None:
        st.info("Complete a Digital Twin run before testing a staffing decision.")
        return
    st.caption(
        "The AFTER run keeps the same scenario, seed, demand, duration, and policy. "
        "Only available agents change."
    )
    after_agents = int(
        st.number_input(
            "AFTER available agents",
            min_value=1,
            value=run.scenario.available_agents + 2,
            step=1,
            key=(
                f"manager_after_agents_{run.scenario.name}_{run.result.seed}_"
                f"{run.scenario.available_agents}"
            ),
        )
    )
    if st.button("RUN BEFORE VS AFTER", type="primary"):
        with st.spinner("Running fair same-seed comparison…"):
            st.session_state.manager_comparison = run_staffing_what_if(run, after_agents)

    comparison: DecisionComparison | None = st.session_state.get("manager_comparison")
    if comparison is None:
        return
    st.info("Simulated effect under this scenario and seed; not a production causal guarantee.")
    st.dataframe(comparison_table(comparison), use_container_width=True, hide_index=True)
    st.caption(
        f"Same seed: {comparison.before.result.seed} · Policy: {comparison.before.policy_mode} · "
        f"Agents: {comparison.before.scenario.available_agents} → "
        f"{comparison.after.scenario.available_agents}"
    )


def _interaction_evidence() -> QualityEvaluationInput | None:
    return st.session_state.get("manager_quality_evidence")


def _render_quality_result() -> None:
    quality = st.session_state.get("manager_quality_result")
    if quality is None:
        return
    st.subheader("Quality Analyst")
    st.write(f"Evaluation status: **{quality.status.value}**")
    st.write(f"Supervisor review required: **{'yes' if quality.requires_supervisor_review else 'no'}**")
    if quality.overall_score is None:
        st.info(
            "Structured quality judge unavailable/offline. No six-dimension or overall scores "
            "are displayed. Deterministic guardrails remain active."
        )
    else:
        st.metric("Overall quality", f"{quality.overall_score:.2f} / 5")
        dimensions = (
            ("Factual accuracy", quality.factual_accuracy),
            ("Procedure adherence", quality.procedure_adherence),
            ("Compliance", quality.compliance),
            ("Relevance", quality.relevance),
            ("Customer satisfaction", quality.customer_satisfaction),
            ("Sentiment handling", quality.sentiment_handling),
        )
        st.dataframe(
            [
                {"Dimension": name, "Score": item.score, "Justification": item.justification}
                for name, item in dimensions
                if item is not None
            ],
            use_container_width=True,
            hide_index=True,
        )
    if quality.flags:
        st.error("Quality flags")
        for flag in quality.flags:
            st.write(f"- **{flag.severity.value.upper()} · {flag.code}** — {flag.explanation}")
    else:
        st.caption("No deterministic quality/compliance flags were observed.")
    if quality.guardrail_observations:
        st.caption("Guardrail observations: " + ", ".join(quality.guardrail_observations))
    if quality.customer_sentiment is not None:
        st.write(f"Customer sentiment: **{quality.customer_sentiment.value}**")


def _interaction_lab() -> None:
    st.header("Interaction Lab")
    st.warning(
        "Selected individual interaction only. High-volume SimPy contacts do not automatically "
        "invoke the Advisor or Quality Analyst."
    )
    modes = ["Offline deterministic demo"]
    if config.GROQ_API_KEY:
        modes.append("Live Groq")
    mode = st.radio("Execution mode", modes, horizontal=True)
    if mode == "Offline deterministic demo":
        st.info("Offline deterministic demo mode — responses are not live LLM output.")
    else:
        st.warning("Live Groq runs only when you press RUN SELECTED INTERACTION.")

    selected = st.selectbox("Demo request", tuple(DEMO_MESSAGES))
    default_customer, default_message = DEMO_MESSAGES[selected]
    c1, c2 = st.columns([1, 3])
    customer_id = c1.text_input(
        "Customer ID", value=default_customer, key=f"lab_customer_{selected}"
    )
    message = c2.text_input(
        "Customer message", value=default_message, key=f"lab_message_{selected}"
    )

    if st.button("RUN SELECTED INTERACTION", type="primary"):
        advisor = (
            CallVerseCustomerAdvisor.from_local_artifact(runner=deterministic_demo_runner)
            if mode == "Offline deterministic demo"
            else CallVerseCustomerAdvisor.from_local_artifact()
        )
        request = SupportRequest(
            request_id=f"LAB-{uuid.uuid4().hex[:10]}",
            customer_id=customer_id,
            intent=RequestIntent.GENERAL,
            customer_message=message,
            arrival_time=0,
        )
        with st.spinner("Running selected interaction…"):
            interaction = advisor.handle_with_trace(request)
        evidence = QualityEvaluationInput.from_advisor_interaction(message, interaction)
        quality = QualityAnalyst().evaluate(evidence)
        st.session_state.manager_interaction = interaction
        st.session_state.manager_interaction_mode = mode
        st.session_state.manager_quality_evidence = evidence
        st.session_state.manager_quality_result = quality
        history = list(st.session_state.get("manager_quality_history", []))
        history.append(quality)
        st.session_state.manager_quality_history = history

    interaction = st.session_state.get("manager_interaction")
    if interaction is None:
        st.info("Run a selected request to inspect classifier and Advisor metadata.")
        return
    st.subheader("Advisor result")
    st.caption(f"Execution mode: {st.session_state.get('manager_interaction_mode')}")
    st.write(interaction.result.response_text)
    st.json(interaction_summary(interaction))

    evidence = _interaction_evidence()
    if config.GROQ_API_KEY and evidence is not None and st.button("RUN LIVE QUALITY JUDGE"):
        with st.spinner("Running structured Groq quality judgment…"):
            quality = QualityAnalyst.with_groq().evaluate(evidence)
        st.session_state.manager_quality_result = quality
        history = list(st.session_state.get("manager_quality_history", []))
        history.append(quality)
        st.session_state.manager_quality_history = history
    _render_quality_result()


def _quality_panel() -> None:
    st.header("Quality")
    st.caption("Individual Advisor evaluations only — never inferred from Digital Twin contacts.")
    history = list(st.session_state.get("manager_quality_history", []))
    summary = session_quality_summary(history)
    if summary is None:
        st.info("No interaction quality evaluations exist in this session.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluated", summary["evaluated_count"])
        c2.metric("Unavailable", summary["unavailable_count"])
        c3.metric("Supervisor review", summary["requires_supervisor_review_count"])
        if summary["average_overall_quality"] is None:
            st.info("No completed structured-judge scores are available; no average is shown.")
        else:
            st.metric("Average overall quality", f"{summary['average_overall_quality']:.2f} / 5")
            st.json(summary["average_by_dimension"])
        st.write("Flags by severity", summary["compliance_flags_by_severity"])
        st.write("Sentiment distribution", summary["sentiment_distribution"])

    benchmark = load_policy_quality_benchmark()
    with st.expander("Quality regression benchmark — policy fixtures, not real customers"):
        st.write(f"Cases defined: **{len(benchmark)}**")
        st.caption(
            "Benchmark scores use deterministic fake-judge fixtures for regression tests and "
            "are never presented as production customer quality."
        )


@st.cache_data
def _load_forecast_view():
    import pandas as pd

    frame = pd.read_csv(FORECAST_SERIES_PATH, parse_dates=["timestamp"])
    history = frame.set_index("timestamp")["contacts"]
    forecast = forecast_next_24h(history, FORECAST_ARTIFACT_PATH)
    return history, forecast, forecast_summary(forecast, history)


def _forecast_panel() -> None:
    st.header("Demand Forecast")
    st.caption(
        "Historical support-demand forecast · next 24 hours · 30-minute intervals. "
        "This is separate from manager-defined scenarios."
    )
    if not FORECAST_SERIES_PATH.is_file() or not FORECAST_ARTIFACT_PATH.is_file():
        st.error(
            "Forecast artifacts are unavailable. Rebuild with "
            "`uv run python -m callverse.forecasting.train`."
        )
        return
    history, forecast, summary = _load_forecast_view()
    first, second, third = st.columns(3)
    first.metric("Predicted next-24h contacts", f"{summary['predicted_total_contacts']:.1f}")
    second.metric("Peak 30-minute slot", summary["peak_timestamp"].strftime("%Y-%m-%d %H:%M"))
    third.metric("Peak predicted contacts", f"{summary['peak_contacts']:.1f}")

    st.subheader("Actual history vs future forecast")
    st.line_chart(
        forecast_chart_rows(history, forecast),
        x="timestamp",
        y=["Actual history", "Forecast"],
        x_label="Historical and forecast timestamp (source time assumption)",
    )
    st.caption(
        f"ACTUAL HISTORY ends {forecast.forecast_origin:%Y-%m-%d %H:%M}; "
        f"FORECAST covers {forecast.points[0].timestamp:%Y-%m-%d %H:%M} through "
        f"{forecast.points[-1].timestamp:%Y-%m-%d %H:%M}. No prediction interval is implied."
    )
    st.info(f"Peak demand is forecast around {summary['peak_timestamp']:%H:%M}.")
    change = summary["next_2h_vs_recent_baseline_percent"]
    if change is not None:
        direction = "above" if change >= 0 else "below"
        st.info(
            f"Forecast demand in the next 2 hours is {abs(change):.1f}% {direction} "
            "the equivalent recent baseline."
        )
    high_periods = summary["high_demand_periods"]
    st.caption(
        "High-demand periods (at least 80% of forecast peak): "
        + (", ".join(item[11:16] for item in high_periods) if high_periods else "none")
    )
    st.warning(
        "Demand signal only: Phase 9 does not recommend staffing levels or alter scenario demand."
    )
    with st.expander("Forecast reproducibility and limitations"):
        st.json(
            {
                "model": forecast.model_name,
                "model_version": forecast.model_version,
                "forecast_origin": forecast.forecast_origin,
                "horizon_slots": forecast.horizon,
                "interval_minutes": forecast.interval_minutes,
                "source": "Technion Anonymous Bank historical contact arrivals (1999)",
                "delivery_signal_used": False,
            }
        )


def render_manager() -> None:
    st.title("CallVerse · Manager Control Room")
    st.caption("Operational Digital Twin metrics and individual interaction quality are separate.")
    tabs = st.tabs(
        [
            "Scenario Studio",
            "Twin Monitor",
            "Compare Decisions",
            "Forecast",
            "Interaction Lab",
            "Quality",
        ]
    )
    with tabs[0]:
        _scenario_studio()
    with tabs[1]:
        _twin_monitor()
    with tabs[2]:
        _compare_decisions()
    with tabs[3]:
        _forecast_panel()
    with tabs[4]:
        _interaction_lab()
    with tabs[5]:
        _quality_panel()
