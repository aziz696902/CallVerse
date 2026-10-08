"""Streamlit composition for the CallVerse Manager Control Room."""

from __future__ import annotations

import json
import uuid
from time import sleep

import streamlit as st

from callverse.customer_advisor import CallVerseCustomerAdvisor
from callverse.customer_advisor_demo import deterministic_demo_runner
from callverse.domain import RequestIntent, SupportRequest
from callverse.forecasting.contracts import DemandForecast
from callverse.forecasting.service import (
    forecast_chart_rows,
    forecast_next_24h,
    forecast_summary,
)
from callverse.quality import QualityAnalyst, QualityEvaluationInput
from callverse.quality.benchmark import load_policy_quality_benchmark
from callverse.scenarios import SCENARIO_PRESETS, get_scenario
from callverse.workforce.dynamic import (
    fair_timeline_rows,
    run_fair_workforce_comparison,
)
from callverse.workforce.manager import (
    build_workforce_plan,
    compare_staffing_strategies,
    default_workforce_config,
    summarize_plan,
    workforce_chart_rows,
)
from helppilot import config

from .comparative_replay import (
    ASSISTED_LABEL,
    BASELINE_LABEL,
    ComparativeReplay,
    DecisionSource,
    ReplaySideFrame,
    build_comparative_replay,
    comparative_slider_key,
    comparison_matches_configuration,
    create_comparison_configuration,
)
from .comparative_story import (
    build_final_manager_story,
    current_frame_differences,
    events_at_frame,
    generate_replay_events,
    large_center_configuration,
    queue_trajectory_rows,
)
from .decision_guidance import (
    PRODUCTION_AB_DISCLAIMER,
    SAME_SEED_EXPLANATION,
    ChangeAssessment,
    DecisionOutcome,
    build_decision_narrative,
    format_metric_delta,
    format_metric_value,
    recommended_decision_widget_state,
)
from .playback import (
    DEFAULT_SPEED,
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
from .scenario_guidance import (
    CenterStatus,
    get_scenario_guide,
    interpret_simulation_result,
    recommended_demo_widget_state,
)
from .twin_replay import (
    build_replay_frame,
    replay_context,
    replay_slider_key,
    waiting_visual,
)
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
from .workforce_visualization import (
    WorkforceVisualFrame,
    build_workforce_visual_frames,
    demand_chart_rows,
    staffing_step_rows,
)
from .workforce_visualization import (
    queue_trajectory_rows as workforce_queue_rows,
)

SCENARIO_NAMES = tuple(scenario.name for scenario in SCENARIO_PRESETS)
PROJECT_ROOT = config.PROJECT_ROOT
FORECAST_SERIES_PATH = PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv"
FORECAST_ARTIFACT_PATH = PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"
WORKFORCE_VALIDATION_PATH = (
    PROJECT_ROOT / "data/processed/workforce/erlang_c_validation.json"
)
RL_MODEL_PATH = PROJECT_ROOT / "models/ppo_workforce/ppo_policy.zip"
RL_METADATA_PATH = PROJECT_ROOT / "models/ppo_workforce/metadata.json"
RL_EVALUATION_PATH = PROJECT_ROOT / "data/processed/rl/policy_evaluation.json"


def rl_artifacts_available(paths=None) -> bool:
    candidates = paths or (RL_MODEL_PATH, RL_METADATA_PATH, RL_EVALUATION_PATH)
    return all(path.is_file() for path in candidates)


DEMO_MESSAGES = {
    "Grounded tracking": ("CUST-1003", "Track my order ORD-5003"),
    "Refund approval": ("CUST-1001", "I want a refund for ORD-5001"),
    "Damaged item fallback": ("CUST-1004", "My item arrived broken"),
    "Low-confidence fallback": ("CUST-1003", "Where is my order?"),
    "Missing order": ("CUST-1003", "Track ORD-9999"),
}


def _render_kpis(run: ManagerRun) -> None:
    cards = kpi_cards(run.result)
    metric_help = {
        "SLA": "Share of served contacts whose service started within the wait target.",
        "Abandonment": "Share of generated contacts that left before service.",
        "Occupancy": "Proportion of available advisor capacity spent busy.",
        "Average handling": "AHT: average handling time for completed contacts.",
    }
    first = st.columns(4)
    second = st.columns(3)
    for column, card in zip((*first, *second), cards):
        column.metric(card.label, card.value, help=metric_help.get(card.label))


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
    interpretation = interpret_simulation_result(run.result)
    status_message = f"CENTER STATUS · {interpretation.status.value}"
    if interpretation.status is CenterStatus.HEALTHY:
        st.success(status_message)
    elif interpretation.status is CenterStatus.UNDER_PRESSURE:
        st.warning(status_message)
    else:
        st.error(status_message)
    st.markdown("**What this means**")
    for reason in interpretation.reasons:
        st.write(f"- {reason}")
    st.markdown("**Operational objectives**")
    outcome_labels = {
        "pass": "PASS",
        "fail": "FAIL",
        "warning": "CAUTION",
        "unavailable": "N/A",
    }
    objective_rows = [
        {
            "Objective": f"{check.metric} {check.objective}",
            "Result": "N/A" if check.actual is None else f"{check.actual:.1%}",
            "Check": outcome_labels[check.outcome],
        }
        for check in interpretation.target_checks
    ]
    st.dataframe(objective_rows, width="stretch", hide_index=True)
    st.info(f"Suggested next step: {interpretation.next_step}")
    st.caption(
        "These are expert-defined V1 managerial targets, not learned thresholds. "
        "The interpretation describes simulated evidence, not guaranteed real-world outcomes."
    )
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
        "Define the operational situation and run the SIMULATED Digital Twin before testing "
        "a staffing decision."
    )

    if st.button("LOAD RECOMMENDED DEMO"):
        for key, value in recommended_demo_widget_state().items():
            st.session_state[key] = value
        st.session_state.manager_comparison = None
        st.rerun()
    st.caption(
        "Recommended demo: Staff Shortage · seed 404 · calibrated mode · 3 agents. "
        "Load the controls, then press RUN DIGITAL TWIN yourself."
    )

    preset_name = st.selectbox(
        "Scenario preset",
        SCENARIO_NAMES,
        key="manager_preset",
        format_func=lambda key: get_scenario_guide(key).title,
    )
    preset = get_scenario(preset_name)
    guide = get_scenario_guide(preset_name)
    with st.container(border=True):
        title_column, risk_column = st.columns([4, 1])
        title_column.subheader(guide.title)
        risk_column.metric("Descriptive risk", guide.risk_level.value)
        st.write(guide.short_description)
        st.markdown(f"**Situation:** {guide.situation}")
        st.markdown(f"**Manager question:** {guide.manager_question}")
        st.markdown("**Operational objectives:** " + " · ".join(guide.objectives))
        st.markdown(f"**Suggested action:** {guide.suggested_action}")
        if guide.demo_recommended:
            st.success(
                "Recommended teaching demo: observe overload, then test higher staffing."
            )
        st.caption(guide.scientific_note)
    c1, c2, c3 = st.columns(3)
    seed_key = f"manager_seed_{preset_name}"
    seed_default = {} if seed_key in st.session_state else {"value": preset.random_seed}
    seed = int(
        c1.number_input(
            "Simulation seed",
            min_value=0,
            step=1,
            key=seed_key,
            **seed_default,
        )
    )
    agents_key = f"manager_agents_{preset_name}"
    agents_default = (
        {} if agents_key in st.session_state else {"value": preset.available_agents}
    )
    agents = int(
        c2.number_input(
            "Available agents",
            min_value=1,
            step=1,
            key=agents_key,
            **agents_default,
        )
    )
    policy_mode = c3.selectbox(
        "Policy mode", ("calibrated", "prototype"), key="manager_policy_mode"
    )
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

    if st.button("RUN DIGITAL TWIN", type="primary", width="stretch"):
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
        st.info(
            "Choose assumptions and press RUN DIGITAL TWIN. The model does not auto-run."
        )
        return
    _render_run_summary(run)
    comparison: DecisionComparison | None = st.session_state.get("manager_comparison")
    st.download_button(
        "Download compact JSON result",
        data=export_json(run, comparison),
        file_name=f"callverse-{run.scenario.name}-{run.result.seed}.json",
        mime="application/json",
    )


def _render_twin_replay(run: ManagerRun) -> None:
    st.subheader("Digital Twin Replay")
    st.caption(
        "The replay visualizes snapshots from the completed simulation. It does not run a "
        "second simulation and does not change the result."
    )
    context = replay_context(run)
    with st.container(border=True):
        context_columns = st.columns(4)
        context_columns[0].metric("Scenario", context.scenario_title)
        context_columns[1].metric("Descriptive risk", context.risk_level)
        context_columns[2].metric("Available agents", context.available_agents)
        context_columns[3].metric("Seed / mode", f"{context.seed} / {context.policy_mode}")
        st.write(f"**Manager question:** {context.manager_question}")

    if not run.result.snapshots:
        st.warning("Replay unavailable because this simulation result contains no snapshots.")
        return

    frame_index = st.slider(
        "Replay time step",
        min_value=0,
        max_value=len(run.result.snapshots) - 1,
        value=0,
        key=replay_slider_key(run),
    )
    frame = build_replay_frame(run, frame_index)
    frame_columns = st.columns(5)
    frame_columns[0].metric(
        "Simulated time", f"{frame.simulated_clock} · min {frame.simulation_minute:g}"
    )
    frame_columns[1].metric("Current queue", frame.queue_size)
    frame_columns[2].metric("Busy agents", frame.busy_agents)
    frame_columns[3].metric("Free agents", frame.free_agents)
    frame_columns[4].metric("Current snapshot pressure", frame.pressure.value)
    st.progress(
        frame.pressure_fraction,
        text=f"Current snapshot pressure: {frame.pressure.value}",
    )
    st.markdown(f"**Waiting:** {waiting_visual(frame)}")
    st.write(
        f"**Staffing:** {frame.busy_agents} busy · {frame.free_agents} free · "
        f"{frame.available_agents} available"
    )
    flow_columns = st.columns(2)
    flow_columns[0].metric("Completed so far", frame.completed_count)
    flow_columns[1].metric("Abandoned so far", frame.abandoned_count)
    st.caption(
        "Current snapshot pressure describes this moment only; cumulative abandonment can "
        "reflect earlier stress in the run. Completed and abandoned are cumulative counters. "
        "Per-snapshot generated contacts, occupancy, SLA, and wait are not stored."
    )


def _mark_comparative_manager_selected() -> None:
    st.session_state.comparative_decision_source = (
        DecisionSource.MANAGER_SELECTED.value
    )


def _mark_comparative_manual_scrub() -> None:
    st.session_state.comparative_manual_scrub_requested = True


def _set_comparative_controls(
    *,
    scenario_name: str,
    seed: int,
    baseline_agents: int,
    assisted_agents: int,
    demand_multiplier: float,
    duration_minutes: float,
    policy_mode: str,
    decision_source: DecisionSource,
) -> None:
    st.session_state.comparative_scenario = scenario_name
    st.session_state.comparative_seed = seed
    st.session_state.comparative_baseline_agents = baseline_agents
    st.session_state.comparative_assisted_agents = assisted_agents
    st.session_state.comparative_demand = demand_multiplier
    st.session_state.comparative_duration = duration_minutes
    st.session_state.comparative_mode = policy_mode
    st.session_state.comparative_decision_source = decision_source.value


def _render_comparative_side(side: ReplaySideFrame) -> None:
    st.markdown(f"#### {side.label}")
    if side.label == BASELINE_LABEL:
        st.caption(
            "Uses the calibrated Digital Twin with the initial fixed staffing level."
        )
    else:
        st.caption(
            "Uses the same Digital Twin and seeded workload, with the tested staffing "
            "decision applied."
        )
    st.caption(f"Fixed staffing for this completed run: {side.available_agents:,} advisors")
    advisors = advisor_markers(side.busy_agents, side.free_agents)
    st.markdown("**Advisor capacity**")
    st.write(" · ".join(f"[{marker}]" for marker in advisors.markers))
    if advisors.hidden_count:
        st.caption(f"+ {advisors.hidden_count} additional advisors")
    waiting = queue_markers(side.queue_size)
    st.markdown("**Waiting contacts**")
    if waiting.markers:
        queue_text = " ".join("●" for _ in waiting.markers)
        if waiting.hidden_count:
            queue_text += f"  + {waiting.hidden_count} more"
        st.write(queue_text)
    else:
        st.write("No contacts waiting")
    first = st.columns(3)
    first[0].metric("Current queue", f"{side.queue_size:,}")
    first[1].metric("Busy advisors", f"{side.busy_agents:,}")
    first[2].metric("Free advisors", f"{side.free_agents:,}")
    second = st.columns(3)
    second[0].metric("Available advisors", f"{side.available_agents:,}")
    second[1].metric("Completed so far", f"{side.completed_so_far:,}")
    second[2].metric("Abandoned so far", f"{side.abandoned_so_far:,}")
    st.metric("Current snapshot pressure", side.snapshot_pressure.value)


def _render_comparative_final_summary(
    replay: ComparativeReplay, *, prominent: bool
) -> None:
    story = build_final_manager_story(replay)

    def render_content() -> None:
        st.caption(
            "Final-run KPIs use the existing Compare Decisions interpretation logic."
        )
        st.dataframe(
            [
                {
                    "Metric": change.label,
                    BASELINE_LABEL: format_metric_value(change, change.before),
                    ASSISTED_LABEL: format_metric_value(change, change.after),
                    "Difference": format_metric_delta(change),
                }
                for change in story.narrative.changes
            ],
            hide_index=True,
            width="stretch",
        )
        st.write(
            f"**{story.narrative.outcome.value}:** {story.narrative.conclusion}"
        )
        st.write(f"**Resource trade-off:** {story.staffing_trade_off}")
        if story.service_effects:
            st.write("**Service effects:** " + " · ".join(story.service_effects))
        st.caption(story.scientific_boundary)
        st.caption(PRODUCTION_AB_DISCLAIMER)

    if prominent:
        st.subheader("FINAL MANAGER SUMMARY")
        render_content()
    else:
        with st.expander("Final result comparison"):
            render_content()


def _comparative_replay_panel(run: ManagerRun | None) -> None:
    st.divider()
    st.subheader("CAPACITY WHAT-IF · COMPARATIVE SIMULATION REPLAY")
    st.write(
        "What happens if the center simply adds staffing? Inspect the fixed baseline and "
        "tested fixed-capacity decision at the same simulated timestamp."
    )
    st.caption(
        "This is a synchronized replay of two completed simulations, not live production "
        "telemetry. CallVerse-assisted means the right-hand simulation applies a decision "
        "supported or tested by CallVerse; it does not mean every model is active at every frame."
    )

    if run is not None:
        run_signature = (
            run.scenario.name,
            run.result.seed,
            run.scenario.available_agents,
            run.scenario.demand_multiplier,
            run.scenario.simulation_duration,
            run.policy_mode,
        )
        if st.session_state.get("comparative_inherited_run") != run_signature:
            _set_comparative_controls(
                scenario_name=run.scenario.name,
                seed=run.result.seed,
                baseline_agents=run.scenario.available_agents,
                assisted_agents=run.scenario.available_agents + 2,
                demand_multiplier=run.scenario.demand_multiplier,
                duration_minutes=run.scenario.simulation_duration,
                policy_mode=run.policy_mode,
                decision_source=DecisionSource.MANAGER_SELECTED,
            )
            st.session_state.comparative_inherited_run = run_signature

    default = run.scenario if run is not None else get_scenario("staff_shortage")
    if "comparative_scenario" not in st.session_state:
        _set_comparative_controls(
            scenario_name=default.name,
            seed=default.random_seed,
            baseline_agents=default.available_agents,
            assisted_agents=default.available_agents + 2,
            demand_multiplier=default.demand_multiplier,
            duration_minutes=default.simulation_duration,
            policy_mode=run.policy_mode if run is not None else "calibrated",
            decision_source=(
                DecisionSource.MANAGER_SELECTED
                if run is not None
                else DecisionSource.RECOMMENDED_DEMO
            ),
        )

    demo_buttons = st.columns(2)
    if demo_buttons[0].button("PREPARE PRIMARY TEACHING DEMO"):
        official = get_scenario("staff_shortage")
        _set_comparative_controls(
            scenario_name=official.name,
            seed=404,
            baseline_agents=3,
            assisted_agents=5,
            demand_multiplier=official.demand_multiplier,
            duration_minutes=official.simulation_duration,
            policy_mode="calibrated",
            decision_source=DecisionSource.RECOMMENDED_DEMO,
        )
        st.session_state.manager_comparative_replay = None
        st.session_state.manager_comparative_playback = None
        st.rerun()
    if demo_buttons[1].button("PREPARE LARGE CENTER STRESS TEST"):
        large = large_center_configuration()
        _set_comparative_controls(
            scenario_name=large.scenario_name,
            seed=large.seed,
            baseline_agents=large.baseline_agents,
            assisted_agents=large.assisted_agents,
            demand_multiplier=large.demand_multiplier,
            duration_minutes=large.duration_minutes,
            policy_mode=large.policy_mode,
            decision_source=large.decision_source,
        )
        st.session_state.manager_comparative_replay = None
        st.session_state.manager_comparative_playback = None
        st.rerun()
    st.caption(
        "PRIMARY — VALIDATED TEACHING DEMO: Staff Shortage · seed 404 · 3 → 5 advisors. "
        "SECONDARY — SCALABILITY DEMO: Large Center Stress Test · seed 404 · "
        "15 → 25 advisors. Preparing either setup does not run a simulation."
    )
    st.caption(
        "Staff Shortage is the primary calibrated demonstration. Large Center Stress Test "
        "scales the same mechanisms for computational and visual scalability; it is not "
        "separately calibrated to a real large call center."
    )

    setup_one = st.columns(4)
    scenario_name = setup_one[0].selectbox(
        "Scenario",
        SCENARIO_NAMES,
        key="comparative_scenario",
        format_func=lambda name: get_scenario_guide(name).title,
        on_change=_mark_comparative_manager_selected,
    )
    seed = int(
        setup_one[1].number_input(
            "Seed",
            min_value=0,
            step=1,
            key="comparative_seed",
            on_change=_mark_comparative_manager_selected,
        )
    )
    baseline_agents = int(
        setup_one[2].number_input(
            "Baseline advisors",
            min_value=1,
            step=1,
            key="comparative_baseline_agents",
            on_change=_mark_comparative_manager_selected,
        )
    )
    assisted_agents = int(
        setup_one[3].number_input(
            "CallVerse-assisted advisors",
            min_value=1,
            step=1,
            key="comparative_assisted_agents",
            on_change=_mark_comparative_manager_selected,
        )
    )
    setup_two = st.columns(3)
    demand_multiplier = float(
        setup_two[0].number_input(
            "Comparison demand multiplier",
            min_value=0.1,
            max_value=20.0,
            step=0.05,
            key="comparative_demand",
            on_change=_mark_comparative_manager_selected,
        )
    )
    duration_minutes = float(
        setup_two[1].number_input(
            "Comparison duration (minutes)",
            min_value=30.0,
            max_value=1440.0,
            step=30.0,
            key="comparative_duration",
            on_change=_mark_comparative_manager_selected,
        )
    )
    policy_mode = setup_two[2].selectbox(
        "Comparison simulator mode",
        ("calibrated", "prototype"),
        key="comparative_mode",
        on_change=_mark_comparative_manager_selected,
    )
    decision_source = DecisionSource(st.session_state.comparative_decision_source)
    comparison_config = create_comparison_configuration(
        scenario_name,
        seed=seed,
        duration_minutes=duration_minutes,
        demand_multiplier=demand_multiplier,
        baseline_agents=baseline_agents,
        assisted_agents=assisted_agents,
        policy_mode=policy_mode,
        decision_source=decision_source,
    )
    st.info(
        f"Both sides use the same {policy_mode} Digital Twin and the same seeded demand "
        "conditions. Only the tested staffing decision changes."
    )
    with st.container(border=True):
        st.markdown("### TESTED DECISION")
        st.metric(
            "Available advisors",
            f"{baseline_agents:,} → {assisted_agents:,}",
        )
        st.write(
            "**Everything else held constant:** scenario, seed, duration, demand, intent and "
            "persona mixes, remaining scenario settings, and simulator mode."
        )
        st.write(f"**Decision source:** {decision_source.value}")
        st.caption(
            "Both sides receive identical seeded demand conditions. Using the same seed "
            "reduces random variation so the staffing decision can be compared under matched "
            "simulated demand."
        )
        st.caption("This is a controlled simulation comparison, not a production A/B test.")
        if decision_source is DecisionSource.SCALABILITY_DEMO:
            st.warning(
                "Scalability demonstration only. Scaling preserves simulator mechanisms but "
                "does not constitute empirical validation at this organization size."
            )

    with st.expander("Changed and held-constant comparison settings"):
        st.write(f"**CHANGED:** Available advisors: {baseline_agents} → {assisted_agents}")
        st.write("**HELD CONSTANT:**")
        st.write(f"- Scenario: {get_scenario_guide(scenario_name).title}")
        st.write(f"- Seed: {seed}")
        st.write(f"- Duration: {duration_minutes:g} simulated minutes")
        st.write(f"- Demand multiplier: {demand_multiplier:g}x")
        st.write(f"- Simulator mode: {policy_mode}")
        st.write("- Intent mix, persona mix, and all other scenario settings")
        st.write(f"**Decision source:** {decision_source.value}")

    if st.button("BUILD COMPARISON", type="primary", width="stretch"):
        with st.spinner("Running the baseline and assisted Digital Twin simulations once…"):
            st.session_state.manager_comparative_replay = build_comparative_replay(
                comparison_config
            )
        initial_state = initial_playback_state(
            len(st.session_state.manager_comparative_replay.frames)
        )
        initial_speed = float(
            st.session_state.get("comparative_playback_speed", DEFAULT_SPEED)
        )
        st.session_state.manager_comparative_playback = set_speed(
            initial_state, initial_speed
        )

    replay: ComparativeReplay | None = st.session_state.get(
        "manager_comparative_replay"
    )
    if replay is None:
        st.info("Build a synchronized comparison first.")
        return
    if not comparison_matches_configuration(replay, comparison_config):
        playback = st.session_state.get("manager_comparative_playback")
        if playback is not None:
            st.session_state.manager_comparative_playback = (
                stop_for_stale_comparison(playback)
            )
        st.warning(
            "The setup changed after this comparison was built. Build the comparison again "
            "before inspecting frames."
        )
        return

    playback: PlaybackState | None = st.session_state.get(
        "manager_comparative_playback"
    )
    if playback is None or playback.frame_index >= len(replay.frames):
        playback = initial_playback_state(len(replay.frames))
        st.session_state.manager_comparative_playback = playback

    slider_key = comparative_slider_key(comparison_config)
    pending_slider_key = f"{slider_key}_pending"
    pending_slider = st.session_state.pop(pending_slider_key, None)
    manual_scrub_requested = st.session_state.pop(
        "comparative_manual_scrub_requested", False
    )
    if pending_slider is not None and not manual_scrub_requested:
        st.session_state[slider_key] = pending_slider
    elif slider_key not in st.session_state:
        st.session_state[slider_key] = playback.frame_index

    controls = st.columns([1, 1, 1, 2])
    if controls[0].button("PLAY", disabled=playback.is_playing or playback.reached_end):
        playback = play(playback, len(replay.frames))
    if controls[1].button("PAUSE", disabled=not playback.is_playing):
        playback = pause(playback)
    if controls[2].button("RESTART"):
        playback = restart(playback, len(replay.frames))
        st.session_state[slider_key] = playback.frame_index
    selected_speed = float(
        controls[3].selectbox(
            "Playback speed",
            SUPPORTED_SPEEDS,
            index=SUPPORTED_SPEEDS.index(playback.speed_multiplier),
            format_func=lambda speed: f"{speed:g}x",
            key="comparative_playback_speed",
            help=(
                "Replay speed controls wall-clock pacing, not simulated time. "
                "At 1x, one 15-minute simulation frame advances every 2 UI seconds."
            ),
        )
    )
    playback = set_speed(playback, selected_speed)
    st.session_state.manager_comparative_playback = playback

    selected_index = st.select_slider(
        "Synchronized simulated time",
        options=tuple(range(len(replay.frames))),
        format_func=lambda index: (
            f"{replay.frames[index].simulated_clock} · "
            f"minute {replay.frames[index].simulation_minute:g}"
        ),
        key=slider_key,
        on_change=_mark_comparative_manual_scrub,
    )
    if selected_index != playback.frame_index:
        playback = scrub(playback, selected_index, len(replay.frames))
        st.session_state.manager_comparative_playback = playback
    frame = current_frame(replay, playback)
    assert frame.baseline.available_agents == replay.comparison.before.result.available_agents
    assert frame.assisted.available_agents == replay.comparison.after.result.available_agents
    cadence = (
        f"{replay.snapshot_cadence_minutes:g} minutes"
        if replay.snapshot_cadence_minutes is not None
        else "variable"
    )
    time_columns = st.columns([1, 2, 1])
    time_columns[1].metric("SIMULATED TIME", frame.simulated_clock)
    time_columns[1].caption(
        f"Frame {frame.index + 1} / {frame.frame_count} · elapsed "
        f"{frame.simulation_minute:g} simulated minutes"
    )
    st.progress(
        frame.index / max(1, frame.frame_count - 1),
        text=(
            f"{replay.frames[0].simulated_clock} → {frame.simulated_clock} → "
            f"{replay.frames[-1].simulated_clock}"
        ),
    )
    st.caption(
        f"Frame {frame.index + 1} of {frame.frame_count} · {frame.simulated_clock} · "
        f"snapshot cadence {cadence} · playback speed {playback.speed_multiplier:g}x"
    )
    if playback.reached_end:
        st.success("Replay complete")
    elif playback.is_playing:
        st.info("Replay playing")
    else:
        st.caption("Replay paused · manual scrubbing is available")

    replay_events = generate_replay_events(replay)
    current_events = events_at_frame(replay_events, frame.index)
    if current_events:
        st.info(
            "Current-frame events: "
            + " · ".join(event.title for event in current_events)
        )
    with st.expander(f"Deterministic timeline events ({len(replay_events)})"):
        st.dataframe(
            [
                {
                    "Time": event.simulated_clock,
                    "Side": event.side.title(),
                    "Event": event.title,
                    "Evidence": event.description,
                }
                for event in replay_events
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Markers are deterministic first occurrences from stored frames. The assisted "
            "staffing level applies from simulation start; no mid-run intervention occurs."
        )

    baseline_column, assisted_column = st.columns(2)
    with baseline_column.container(border=True):
        _render_comparative_side(frame.baseline)
    with assisted_column.container(border=True):
        _render_comparative_side(frame.assisted)

    st.markdown(f"**AT {frame.simulated_clock} SIMULATED TIME**")
    differences = current_frame_differences(frame)
    st.dataframe(
        [
            {
                "Metric": difference.metric,
                BASELINE_LABEL: (
                    f"{difference.baseline:,} / {frame.baseline.available_agents:,}"
                    if difference.metric == "Busy advisors"
                    else f"{difference.baseline:,}"
                ),
                ASSISTED_LABEL: (
                    f"{difference.assisted:,} / {frame.assisted.available_agents:,}"
                    if difference.metric == "Busy advisors"
                    else f"{difference.assisted:,}"
                ),
                "Assisted difference": f"{difference.assisted_delta:+,}",
                "Evidence type": difference.semantics,
            }
            for difference in differences
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Current snapshot pressure describes this moment; cumulative outcomes include earlier "
        "stress in the run. Mid-run differences are descriptive simulated evidence, not a "
        "production causal estimate."
    )
    st.markdown("**Queue size over simulated time**")
    st.line_chart(
        queue_trajectory_rows(replay),
        x="simulation_time",
        y=[BASELINE_LABEL, ASSISTED_LABEL],
        x_label="Simulated minutes",
    )
    st.caption(
        "Every point is an actual synchronized replay snapshot; no smoothing or "
        "interpolation is applied."
    )
    _render_comparative_final_summary(replay, prominent=playback.reached_end)

    if playback.is_playing:
        sleep(seconds_per_frame(playback.speed_multiplier))
        playback = advance(playback, len(replay.frames))
        st.session_state.manager_comparative_playback = playback
        st.session_state[pending_slider_key] = playback.frame_index
        st.rerun()


def _twin_monitor() -> None:
    st.header("Twin Monitor")
    st.caption(
        "SIMULATED evidence · Observe queue, staffing, and contact flow from the completed "
        "Digital Twin run."
    )
    run: ManagerRun | None = st.session_state.get("manager_run")
    if run is None:
        st.info("Run a Digital Twin scenario to unlock the replay.")
    else:
        _render_twin_replay(run)
        st.divider()
        st.subheader("Full-run monitoring")
        rows = timeline_rows(run.result)
        st.subheader("Queue size over simulated time")
        st.line_chart(
            rows, x="simulation_time", y="queue_size", x_label="Simulated minutes"
        )
        st.subheader("Busy agents over simulated time")
        st.line_chart(
            rows, x="simulation_time", y="busy_agents", x_label="Simulated minutes"
        )
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
    _comparative_replay_panel(run)


def _compare_decisions() -> None:
    st.header("Compare Decisions")
    st.caption(
        "CAPACITY WHAT-IF · SIMULATED same-seed comparison · Test what happens when one "
        "fixed staffing choice changes. "
        "Workforce separately answers what Erlang-C analytically recommends."
    )
    run: ManagerRun | None = st.session_state.get("manager_run")
    if run is None:
        st.info("Complete a Digital Twin run before testing a staffing decision.")
        return

    before_status = interpret_simulation_result(run.result).status.value.lower()
    guide = get_scenario_guide(run.scenario.name)
    st.info(
        f"Inherited from Scenario Studio: **{guide.title}** · seed **{run.result.seed}** · "
        f"**{run.policy_mode}** mode · **{run.scenario.available_agents} agents**."
    )
    st.subheader("1 · PROBLEM")
    st.write(
        f"{guide.title} with {run.scenario.available_agents} agents currently shows a "
        f"**{before_status}** simulated center."
    )
    st.caption(SAME_SEED_EXPLANATION)
    st.caption(PRODUCTION_AB_DISCLAIMER)

    comparison_key = (
        f"manager_after_agents_{run.scenario.name}_{run.result.seed}_"
        f"{run.scenario.available_agents}"
    )
    official_demo_context = (
        run.scenario.name == "staff_shortage"
        and run.result.seed == 404
        and run.scenario.available_agents == 3
        and run.policy_mode == "calibrated"
    )
    if official_demo_context and st.button("PREPARE RECOMMENDED DECISION TEST"):
        for key, value in recommended_decision_widget_state(comparison_key).items():
            st.session_state[key] = value
        st.session_state.manager_comparison = None
        st.rerun()

    st.subheader("2 · PROPOSED ACTION")
    after_default = (
        {}
        if comparison_key in st.session_state
        else {"value": run.scenario.available_agents + 2}
    )
    after_agents = int(
        st.number_input(
            "AFTER available agents",
            min_value=1,
            step=1,
            key=comparison_key,
            **after_default,
        )
    )
    st.write(
        f"Test available agents: **{run.scenario.available_agents} → {after_agents}**. "
        "No other simulation input changes."
    )
    with st.expander("Changed and held-constant configuration", expanded=True):
        st.write(
            f"**CHANGED:** Available agents: {run.scenario.available_agents} → {after_agents}"
        )
        st.write("**HELD CONSTANT:**")
        st.write(f"- Scenario: {guide.title}")
        st.write(f"- Seed: {run.result.seed}")
        st.write(f"- Simulator mode: {run.policy_mode}")
        st.write(f"- Duration: {run.scenario.simulation_duration:g} simulated minutes")
        st.write(f"- Demand multiplier: {run.scenario.demand_multiplier:g}x")
        st.write("- Persona mix, request mix, and all other scenario settings")
    if st.button("RUN COMPARISON", type="primary"):
        with st.spinner("Running fair same-seed comparison…"):
            st.session_state.manager_comparison = run_staffing_what_if(
                run, after_agents
            )

    comparison: DecisionComparison | None = st.session_state.get("manager_comparison")
    if comparison is None:
        st.info("Prepare the staffing idea, then press RUN COMPARISON. Nothing auto-runs.")
        return

    narrative = build_decision_narrative(comparison)
    st.subheader("3 · SIMULATED EFFECT")
    first_row = st.columns(3)
    second_row = st.columns(3)
    for column, change in zip((*first_row, *second_row), narrative.changes, strict=True):
        delta_color = "off"
        if change.assessment is not ChangeAssessment.NEUTRAL:
            delta_color = "normal" if change.preference == "higher" else "inverse"
        column.metric(
            change.label,
            (
                f"{format_metric_value(change, change.before)} → "
                f"{format_metric_value(change, change.after)}"
            ),
            delta=format_metric_delta(change),
            delta_color=delta_color,
            help=(
                "Contacts still waiting when the simulation ends."
                if change.key == "final_backlog"
                else None
            ),
        )

    st.markdown("**Operational target checks**")
    target_labels = {
        "pass": "PASS",
        "fail": "FAIL",
        "warning": "CAUTION",
        "unavailable": "N/A",
    }
    st.dataframe(
        [
            {
                "Objective": f"{target.metric} {target.objective}",
                "BEFORE": target_labels[target.before],
                "AFTER": target_labels[target.after],
            }
            for target in narrative.targets
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Generated contacts are simulated arrivals. Completed contacts are those served within "
        "the horizon; abandonment and remaining work can make completed totals differ. A higher "
        "completed count alone does not prove a better configuration."
    )
    with st.expander("Detailed KPI table and reproducibility"):
        st.dataframe(comparison_table(comparison), width="stretch", hide_index=True)
        st.write(f"**CHANGED:** {narrative.changed_parameter}")
        st.write("**HELD CONSTANT:** " + " · ".join(narrative.held_constant))

    st.subheader("4 · MANAGER CONCLUSION")
    outcome_message = f"{narrative.outcome.value}: {narrative.conclusion}"
    if narrative.outcome in {
        DecisionOutcome.STRONGLY_IMPROVED,
        DecisionOutcome.IMPROVED,
    }:
        st.success(outcome_message)
    elif narrative.outcome is DecisionOutcome.WORSENED:
        st.error(outcome_message)
    else:
        st.warning(outcome_message)
    st.write(narrative.trade_off)
    st.info(narrative.next_step)
    st.caption(
        "Compare Decisions manually tests one staffing idea in the Digital Twin. Workforce "
        "separately provides an Erlang-C analytical staffing recommendation; neither claims "
        "this tested configuration is optimal."
    )


def _interaction_evidence() -> QualityEvaluationInput | None:
    return st.session_state.get("manager_quality_evidence")


def _render_quality_result() -> None:
    quality = st.session_state.get("manager_quality_result")
    if quality is None:
        return
    st.subheader("Quality Analyst")
    st.write(f"Evaluation status: **{quality.status.value}**")
    st.write(
        f"Supervisor review required: **{'yes' if quality.requires_supervisor_review else 'no'}**"
    )
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
                {
                    "Dimension": name,
                    "Score": item.score,
                    "Justification": item.justification,
                }
                for name, item in dimensions
                if item is not None
            ],
            width="stretch",
            hide_index=True,
        )
    if quality.flags:
        st.error("Quality flags")
        for flag in quality.flags:
            st.write(
                f"- **{flag.severity.value.upper()} · {flag.code}** — {flag.explanation}"
            )
    else:
        st.caption("No deterministic quality/compliance flags were observed.")
    if quality.guardrail_observations:
        st.caption(
            "Guardrail observations: " + ", ".join(quality.guardrail_observations)
        )
    if quality.customer_sentiment is not None:
        st.write(f"Customer sentiment: **{quality.customer_sentiment.value}**")


def _interaction_lab() -> None:
    st.header("Interaction Lab")
    st.write(
        "Demonstrate one Advisor interaction: the classifier identifies the request, "
        "structured tools retrieve order/customer facts, RAG retrieves policies and "
        "procedures, and an LLM writes the response when the live provider is selected."
    )
    st.caption(
        "Order and customer facts come from structured tools. Policies and procedures come "
        "from RAG. Unsupported or unsafe cases escalate; sensitive actions may require approval."
    )
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
        if not config.GROQ_API_KEY:
            st.error(
                "Live LLM provider unavailable because GROQ_API_KEY is not configured. "
                "No fake live result will be substituted."
            )
    else:
        st.warning(
            "LIVE / LLM-ASSISTED Groq execution is available and runs only when you press "
            "RUN SELECTED INTERACTION."
        )

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
        st.session_state.manager_interaction_error = None
        advisor = (
            CallVerseCustomerAdvisor.from_local_artifact(
                runner=deterministic_demo_runner
            )
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
            try:
                interaction = advisor.handle_with_trace(request)
            except Exception as exc:  # noqa: BLE001 - provider failures must fail visibly
                st.error(
                    "Live provider unavailable; no deterministic response was substituted. "
                    f"Error type: {type(exc).__name__}."
                )
                return
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
    if (
        config.GROQ_API_KEY
        and evidence is not None
        and st.button("RUN LIVE QUALITY JUDGE")
    ):
        with st.spinner("Running structured Groq quality judgment…"):
            quality = QualityAnalyst.with_groq().evaluate(evidence)
        st.session_state.manager_quality_result = quality
        history = list(st.session_state.get("manager_quality_history", []))
        history.append(quality)
        st.session_state.manager_quality_history = history
    _render_quality_result()


def _quality_panel() -> None:
    st.header("Quality")
    st.caption(
        "Evaluate an individual Advisor response for quality and compliance. The deterministic "
        "layer enforces hard safety/compliance flags; the optional LLM layer judges nuanced "
        "dimensions such as relevance and sentiment handling. LLM scores are not human ground truth."
    )
    history = list(st.session_state.get("manager_quality_history", []))
    summary = session_quality_summary(history)
    if summary is None:
        st.info(
            "No interaction quality evaluation is available. Run an Interaction Lab request "
            "first; deterministic safeguards remain active even without a live LLM judge."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluated", summary["evaluated_count"])
        c2.metric("Unavailable", summary["unavailable_count"])
        c3.metric("Supervisor review", summary["requires_supervisor_review_count"])
        if summary["average_overall_quality"] is None:
            st.info(
                "No completed structured-judge scores are available; no average is shown."
            )
        else:
            st.metric(
                "Average overall quality",
                f"{summary['average_overall_quality']:.2f} / 5",
            )
            st.json(summary["average_by_dimension"])
        st.write("Flags by severity", summary["compliance_flags_by_severity"])
        st.write("Sentiment distribution", summary["sentiment_distribution"])

    benchmark = load_policy_quality_benchmark()
    with st.expander(
        "Quality regression benchmark — policy fixtures, not real customers"
    ):
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
    st.info(
        "HISTORICAL ML FORECAST: selected LightGBM Poisson model forecasting contact demand "
        "for the next 48 half-hour slots (24 hours). It is not a live weather forecast and "
        "does not predict customer satisfaction."
    )
    st.header("Demand Forecast")
    st.caption(
        "Estimate upcoming contact volume from historical Technion generic contact-center data. "
        "This evidence is separate from manager-defined simulated scenarios."
    )
    if not FORECAST_SERIES_PATH.is_file() or not FORECAST_ARTIFACT_PATH.is_file():
        st.error(
            "Forecast artifacts are unavailable. Rebuild with "
            "`uv run python -m callverse.forecasting.train`."
        )
        return
    history, forecast, summary = _load_forecast_view()
    first, second, third = st.columns(3)
    first.metric(
        "Predicted next-24h contacts", f"{summary['predicted_total_contacts']:.1f}"
    )
    second.metric(
        "Peak 30-minute slot", summary["peak_timestamp"].strftime("%Y-%m-%d %H:%M")
    )
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


def _legacy_fair_workforce_comparison_panel(forecast: DemandForecast) -> None:
    st.divider()
    st.subheader("FAIR WORKFORCE COMPARISON")
    st.caption(
        "WORKFORCE INTELLIGENCE · Can forecast-aware scheduling allocate the same or "
        "lower total staffing budget better over time? This is separate from the fixed "
        "3→5 CAPACITY WHAT-IF."
    )
    st.write(
        "**Fixed baseline:** 2 advisors throughout 24 hours.  "
        "**CallVerse dynamic:** the existing default Forecast → Erlang-C plan, applied "
        "at 30-minute boundaries."
    )
    st.caption(
        "The experiment is predefined: seed 404, current forecast artifact, default "
        "80% service target, 85% occupancy cap, 10% forecast buffer, and no PPO."
    )
    if st.button("RUN FAIR WORKFORCE COMPARISON", type="primary", width="stretch"):
        with st.spinner("Running matched fixed and scheduled 24-hour simulations…"):
            st.session_state.manager_fair_workforce_comparison = (
                run_fair_workforce_comparison(forecast)
            )

    comparison = st.session_state.get("manager_fair_workforce_comparison")
    if comparison is None:
        st.info(
            "Run the comparison to evaluate actual Digital Twin outcomes. Nothing auto-runs."
        )
        return

    baseline_hours = comparison.baseline_schedule.total_agent_hours
    callverse_hours = comparison.callverse_schedule.total_agent_hours
    budget = st.columns(4)
    budget[0].metric("Baseline agent-hours", f"{baseline_hours:.1f}")
    budget[1].metric("CallVerse agent-hours", f"{callverse_hours:.1f}")
    budget[2].metric(
        "Budget difference", f"{comparison.resource_budget_delta_hours:+.1f} h"
    )
    budget[3].metric(
        "CallVerse min / avg / max",
        (
            f"{comparison.callverse_schedule.minimum_advisors} / "
            f"{comparison.callverse_schedule.average_advisors:.2f} / "
            f"{comparison.callverse_schedule.maximum_advisors}"
        ),
    )
    if comparison.fairness_satisfied:
        st.success(f"RESOURCE BUDGET PASSES · {comparison.outcome.value}")
    else:
        st.error(f"RESOURCE BUDGET FAILS · {comparison.outcome.value}")

    baseline = comparison.baseline_result
    callverse = comparison.callverse_result
    baseline_backlog = baseline.snapshots[-1].queue_size
    callverse_backlog = callverse.snapshots[-1].queue_size
    st.markdown("**SERVICE OUTCOME · actual simulated results**")
    st.dataframe(
        [
            {
                "Metric": "Generated contacts",
                "Fixed baseline": f"{baseline.counts.generated:,}",
                "CallVerse dynamic": f"{callverse.counts.generated:,}",
                "Difference": f"{callverse.counts.generated - baseline.counts.generated:+,}",
            },
            {
                "Metric": "Completed contacts",
                "Fixed baseline": f"{baseline.counts.completed:,}",
                "CallVerse dynamic": f"{callverse.counts.completed:,}",
                "Difference": f"{callverse.counts.completed - baseline.counts.completed:+,}",
            },
            {
                "Metric": "SLA",
                "Fixed baseline": f"{baseline.kpis.sla:.2%}",
                "CallVerse dynamic": f"{callverse.kpis.sla:.2%}",
                "Difference": f"{(callverse.kpis.sla - baseline.kpis.sla) * 100:+.2f} pp",
            },
            {
                "Metric": "Abandonment",
                "Fixed baseline": f"{baseline.kpis.abandonment_rate:.2%}",
                "CallVerse dynamic": f"{callverse.kpis.abandonment_rate:.2%}",
                "Difference": (
                    f"{(callverse.kpis.abandonment_rate - baseline.kpis.abandonment_rate) * 100:+.2f} pp"
                ),
            },
            {
                "Metric": "Average wait",
                "Fixed baseline": f"{baseline.kpis.average_waiting_time:.2f} min",
                "CallVerse dynamic": f"{callverse.kpis.average_waiting_time:.2f} min",
                "Difference": (
                    f"{callverse.kpis.average_waiting_time - baseline.kpis.average_waiting_time:+.2f} min"
                ),
            },
            {
                "Metric": "Occupancy",
                "Fixed baseline": f"{baseline.kpis.occupancy:.2%}",
                "CallVerse dynamic": f"{callverse.kpis.occupancy:.2%}",
                "Difference": (
                    f"{(callverse.kpis.occupancy - baseline.kpis.occupancy) * 100:+.2f} pp"
                ),
            },
            {
                "Metric": "Final backlog",
                "Fixed baseline": f"{baseline_backlog:,}",
                "CallVerse dynamic": f"{callverse_backlog:,}",
                "Difference": f"{callverse_backlog - baseline_backlog:+,}",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Same forecast-shaped realized demand, seed, customer mix, service distributions, "
        "and patience distributions. Only fixed versus scheduled capacity differs."
    )

    with st.expander("30-minute schedule and operational evidence"):
        st.dataframe(
            fair_timeline_rows(comparison, forecast),
            hide_index=True,
            width="stretch",
        )
    with st.expander(
        f"WORKFORCE PLAN CHANGES ({len(comparison.staffing_events)})"
    ):
        st.dataframe(
            [
                {
                    "Time": event.timestamp.strftime("%H:%M"),
                    "Previous advisors": event.previous_advisors,
                    "Scheduled advisors": event.advisors,
                    "Event": event.title,
                }
                for event in comparison.staffing_events
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "These are predefined workforce-plan changes, not manager manual interventions. "
            "Staffing reductions never interrupt contacts already in service."
        )
    st.warning(
        "Simulation evidence only. The schedule does not model named shifts, skills, labor "
        "law, overtime, or a production optimum."
    )


def _render_workforce_floor_side(title: str, snapshot) -> None:
    st.markdown(f"#### {title}")
    st.caption("CURRENT CAPACITY AND PRESSURE")
    stats = st.columns(3)
    stats[0].metric("Busy", snapshot.busy_agents)
    stats[1].metric("Free", snapshot.free_agents)
    stats[2].metric("Queue", snapshot.queue_size)
    st.caption(
        f"Available {snapshot.available_agents} · overhang busy "
        f"{snapshot.overhang_busy_agents}"
    )
    advisors = advisor_markers(snapshot.busy_agents, snapshot.free_agents, cap=10)
    waiting = queue_markers(snapshot.queue_size, cap=10)
    st.write(
        "Advisors: "
        + (" · ".join(advisors.markers) or "none")
        + (f" · +{advisors.hidden_count} hidden" if advisors.hidden_count else "")
    )
    st.write(
        "Queue: "
        + (" · ".join(waiting.markers) or "empty")
        + (f" · +{waiting.hidden_count} hidden" if waiting.hidden_count else "")
    )
    totals = st.columns(2)
    st.caption("CUMULATIVE FLOW")
    totals[0].metric("Completed", snapshot.completed_count)
    totals[1].metric("Abandoned", snapshot.abandoned_count)


def _render_workforce_final_result(comparison) -> None:
    baseline = comparison.baseline_result
    callverse = comparison.callverse_result
    fixed_label = f"Fixed · {comparison.baseline_schedule.total_agent_hours:.1f} h"
    dynamic_label = (
        f"CallVerse · {comparison.callverse_schedule.total_agent_hours:.1f} h"
    )
    rows = (
        (
            "Completed",
            f"{baseline.counts.completed:d}",
            f"{callverse.counts.completed:d}",
            f"{callverse.counts.completed - baseline.counts.completed:+d}",
        ),
        (
            "SLA",
            f"{baseline.kpis.sla:.2%}",
            f"{callverse.kpis.sla:.2%}",
            f"{(callverse.kpis.sla - baseline.kpis.sla) * 100:+.2f} pp",
        ),
        (
            "Abandonment",
            f"{baseline.kpis.abandonment_rate:.2%}",
            f"{callverse.kpis.abandonment_rate:.2%}",
            (
                f"{(callverse.kpis.abandonment_rate - baseline.kpis.abandonment_rate) * 100:+.2f} pp"
            ),
        ),
        (
            "Average wait",
            f"{baseline.kpis.average_waiting_time:.2f} min",
            f"{callverse.kpis.average_waiting_time:.2f} min",
            (
                f"{callverse.kpis.average_waiting_time - baseline.kpis.average_waiting_time:+.2f} min"
            ),
        ),
        (
            "Occupancy",
            f"{baseline.kpis.occupancy:.2%}",
            f"{callverse.kpis.occupancy:.2%}",
            f"{(callverse.kpis.occupancy - baseline.kpis.occupancy) * 100:+.2f} pp",
        ),
        (
            "Final backlog",
            f"{baseline.snapshots[-1].queue_size:d}",
            f"{callverse.snapshots[-1].queue_size:d}",
            (
                f"{callverse.snapshots[-1].queue_size - baseline.snapshots[-1].queue_size:+d}"
            ),
        ),
    )
    st.markdown("## FINAL 24-HOUR RESULT · ACTUAL SIMULATED OUTCOMES")
    st.success(comparison.outcome.value)
    st.dataframe(
        [
            {
                "Metric": metric,
                fixed_label: fixed,
                dynamic_label: dynamic,
                "Difference": delta,
            }
            for metric, fixed, dynamic, delta in rows
        ],
        hide_index=True,
        width="stretch",
    )
    saving_percent = (
        comparison.baseline_schedule.total_agent_hours
        - comparison.callverse_schedule.total_agent_hours
    ) / comparison.baseline_schedule.total_agent_hours
    st.info(
        f"CallVerse achieved better simulated service with {saving_percent:.2%} fewer "
        "total agent-hours by reallocating capacity toward higher-demand periods."
    )
    st.caption(
        "This is evidence from the calibrated simulation experiment, not a production "
        "optimality guarantee."
    )


def _fair_workforce_comparison_panel(forecast: DemandForecast) -> None:
    st.divider()
    st.subheader("WORKFORCE INTELLIGENCE COMPARISON")
    st.markdown("**Same demand. Lower staffing budget. Smarter allocation.**")
    st.caption(
        "SIMULATED · A synchronized 24-hour replay of fixed capacity versus the "
        "predefined Forecast → Erlang-C schedule. This is separate from the fixed "
        "3→5 CAPACITY WHAT-IF."
    )
    st.write(
        "Can CallVerse allocate a lower staffing budget more effectively over time?"
    )
    st.caption(
        "Predefined experiment: seed 404, current forecast artifact, 80% service target, "
        "85% occupancy cap, 10% forecast buffer, and no PPO."
    )
    signature = (
        forecast.forecast_origin.isoformat(),
        forecast.model_name,
        forecast.model_version,
        tuple(point.predicted_contacts for point in forecast.points),
    )
    if st.button("LOAD WORKFORCE INTELLIGENCE DEMO", width="stretch"):
        st.session_state.manager_workforce_demo_prepared = signature
        for key in (
            "manager_fair_workforce_comparison",
            "manager_workforce_visual_frames",
            "manager_workforce_playback",
            "manager_workforce_comparison_signature",
            "manager_workforce_replay_position",
        ):
            st.session_state.pop(key, None)

    prepared_signature = st.session_state.get("manager_workforce_demo_prepared")
    if prepared_signature != signature:
        playback = st.session_state.get("manager_workforce_playback")
        if playback is not None:
            st.session_state.manager_workforce_playback = stop_for_stale_comparison(
                playback
            )
        message = (
            "Load the defense demo to prepare its fixed inputs. Loading does not run "
            "either simulation or reveal an outcome."
            if prepared_signature is None
            else "The forecast input changed. Reload the demo before running or replaying."
        )
        st.info(message)
        return

    with st.container(border=True):
        st.markdown("### PREDEFINED FAIR TEST")
        st.write(
            "**Fixed baseline:** 2 advisors throughout 24 hours.  "
            "**CallVerse dynamic:** the existing default Forecast → Erlang-C plan at "
            "30-minute boundaries."
        )
        st.write(
            "**Held constant:** forecast-shaped demand, seed 404, customer mix, service "
            "and patience distributions, SLA definition, and simulator."
        )
        st.caption("Only the staffing policy changes. Nothing has run yet.")

    if st.button("RUN FAIR COMPARISON", type="primary", width="stretch"):
        with st.spinner("Running the matched fixed and scheduled simulations once…"):
            comparison = run_fair_workforce_comparison(forecast)
            frames = build_workforce_visual_frames(comparison, forecast)
            st.session_state.manager_fair_workforce_comparison = comparison
            st.session_state.manager_workforce_visual_frames = frames
            st.session_state.manager_workforce_playback = initial_playback_state(
                len(frames)
            )
            st.session_state.manager_workforce_comparison_signature = signature

    comparison = st.session_state.get("manager_fair_workforce_comparison")
    if comparison is None:
        st.info("Inputs are ready. Run the fair comparison; nothing auto-runs.")
        return
    if st.session_state.get("manager_workforce_comparison_signature") != signature:
        playback = st.session_state.get("manager_workforce_playback")
        if playback is not None:
            st.session_state.manager_workforce_playback = stop_for_stale_comparison(
                playback
            )
        st.warning("This stored result is stale. Reload and rerun the fair comparison.")
        return

    frames: tuple[WorkforceVisualFrame, ...] | None = st.session_state.get(
        "manager_workforce_visual_frames"
    )
    if not frames:
        frames = build_workforce_visual_frames(comparison, forecast)
        st.session_state.manager_workforce_visual_frames = frames
    baseline_hours = comparison.baseline_schedule.total_agent_hours
    callverse_hours = comparison.callverse_schedule.total_agent_hours
    saving_percent = (baseline_hours - callverse_hours) / baseline_hours
    st.success(
        f"FAIRNESS PASSES · {comparison.baseline_result.counts.generated} contacts on both "
        f"sides · seed {comparison.seed} · {baseline_hours:.1f} vs {callverse_hours:.1f} "
        f"agent-hours · {comparison.resource_budget_delta_hours:+.1f} agent-hours · "
        f"{saving_percent:.2%} fewer agent-hours"
    )
    st.caption(
        "Same forecast-shaped realized demand, seed, customer mix, service distributions, "
        "and patience distributions. Only fixed versus scheduled capacity differs."
    )

    playback: PlaybackState | None = st.session_state.get(
        "manager_workforce_playback"
    )
    if playback is None or playback.frame_index >= len(frames):
        playback = initial_playback_state(len(frames))
    slider_key = "manager_workforce_replay_position"
    pending = st.session_state.pop("manager_workforce_replay_pending", None)
    manual = st.session_state.pop("manager_workforce_manual_scrub", False)
    if pending is not None and not manual:
        st.session_state[slider_key] = pending
    elif slider_key not in st.session_state:
        st.session_state[slider_key] = playback.frame_index

    controls = st.columns([1, 1, 1, 2])
    if controls[0].button(
        "PLAY",
        key="workforce_play",
        disabled=playback.is_playing or playback.reached_end,
    ):
        playback = play(playback, len(frames))
    if controls[1].button(
        "PAUSE", key="workforce_pause", disabled=not playback.is_playing
    ):
        playback = pause(playback)
    if controls[2].button("RESTART", key="workforce_restart"):
        playback = restart(playback, len(frames))
        st.session_state[slider_key] = 0
    selected_speed = float(
        controls[3].selectbox(
            "Playback speed",
            SUPPORTED_SPEEDS,
            index=SUPPORTED_SPEEDS.index(playback.speed_multiplier),
            format_func=lambda speed: f"{speed:g}x",
            key="workforce_playback_speed",
            help="Wall-clock pacing only; simulations are not rerun during playback.",
        )
    )
    playback = set_speed(playback, selected_speed)
    st.session_state.manager_workforce_playback = playback

    selected_index = st.select_slider(
        "Synchronized 24-hour simulated time",
        options=tuple(range(len(frames))),
        format_func=lambda index: (
            f"{frames[index].simulated_clock} · frame {index + 1}/{len(frames)}"
        ),
        key=slider_key,
        on_change=lambda: st.session_state.update(
            manager_workforce_manual_scrub=True
        ),
    )
    if selected_index != playback.frame_index:
        playback = scrub(playback, selected_index, len(frames))
        st.session_state.manager_workforce_playback = playback
    frame = frames[playback.frame_index]

    clock_columns = st.columns([1, 2, 1])
    clock_columns[1].metric("SIMULATED CLOCK", frame.simulated_clock)
    clock_columns[1].caption(f"Frame {frame.index + 1} / {frame.frame_count}")
    st.progress(
        frame.index / max(1, frame.frame_count - 1),
        text=f"00:00 → {frame.simulated_clock} → 24:00",
    )
    if playback.reached_end:
        st.success("24-hour Workforce Intelligence replay complete")
    elif playback.is_playing:
        st.info(f"Replay playing at {playback.speed_multiplier:g}x")
    else:
        st.caption("Replay paused · use the synchronized scrubber to inspect any frame.")

    st.markdown("### KNOWN DEMAND AND STAFFING PLAN")
    peak_point = max(forecast.points, key=lambda point: point.predicted_contacts)
    peak_columns = st.columns(2)
    peak_columns[0].metric("FORECAST PEAK", peak_point.timestamp.strftime("%H:%M"))
    peak_columns[1].metric(
        "Predicted contacts / slot", f"{peak_point.predicted_contacts:.2f}"
    )
    st.caption(
        f"Current position: {frame.simulated_clock} · forecast slot demand "
        f"{frame.forecast_contacts:.2f} contacts"
    )
    st.markdown("**Forecast demand · 48 half-hour slots**")
    st.line_chart(
        demand_chart_rows(forecast, frame.simulation_minute),
        x="simulation_minute",
        y=["Forecast contacts", "Current time"],
        x_label="Simulated minute (00:00–24:00)",
    )
    st.markdown("**Staffing schedule · exact 30-minute steps**")
    st.line_chart(
        staffing_step_rows(comparison, frame.simulation_minute),
        x="simulation_minute",
        y=["Fixed 2 advisors", "CallVerse dynamic"],
        x_label="Simulated minute (00:00–24:00)",
    )
    st.info(frame.resource_explanation)
    if frame.staffing_event:
        st.success(f"SCHEDULE EVENT · {frame.staffing_event}")

    resource_columns = st.columns(2)
    resource_columns[0].metric(
        "Fixed cumulative agent-hours", f"{frame.baseline_agent_hours:.2f}"
    )
    resource_columns[1].metric(
        "CallVerse cumulative agent-hours", f"{frame.callverse_agent_hours:.2f}"
    )

    st.markdown("### LIVE OPERATIONAL FLOOR · STORED SNAPSHOTS")
    fixed_column, dynamic_column = st.columns(2)
    with fixed_column.container(border=True):
        _render_workforce_floor_side("FIXED · 2 ADVISORS", frame.baseline)
    with dynamic_column.container(border=True):
        _render_workforce_floor_side("CALLVERSE · DYNAMIC", frame.callverse)

    st.markdown("**Queue trajectory revealed through the current frame**")
    st.line_chart(
        workforce_queue_rows(frames, frame.index),
        x="simulation_minute",
        y=["Fixed 2 advisors", "CallVerse dynamic"],
        x_label="Simulated minute",
    )
    st.caption(
        "Actual stored snapshots only; no smoothing, estimation, or replay-time simulation."
    )
    with st.expander(
        f"PREDEFINED STAFFING EVENTS ({len(comparison.staffing_events)})"
    ):
        st.dataframe(
            [
                {
                    "Time": (
                        f"{int(event.simulation_minute // 60):02d}:"
                        f"{int(event.simulation_minute % 60):02d}"
                    ),
                    "Change": f"{event.previous_advisors} → {event.advisors}",
                    "Evidence": event.title,
                }
                for event in comparison.staffing_events
            ],
            hide_index=True,
            width="stretch",
        )

    if playback.reached_end:
        _render_workforce_final_result(comparison)
    else:
        st.caption("Final service outcomes unlock only when the replay reaches 24:00.")
    st.warning(
        "Simulation evidence only. The schedule does not model named shifts, skills, labor "
        "law, overtime, or a production optimum."
    )

    if playback.is_playing:
        sleep(seconds_per_frame(playback.speed_multiplier))
        playback = advance(playback, len(frames))
        st.session_state.manager_workforce_playback = playback
        st.session_state.manager_workforce_replay_pending = playback.frame_index
        st.rerun()


def _workforce_panel() -> None:
    st.info(
        "ANALYTICAL ERLANG-C BASELINE + DYNAMIC TWIN EVALUATION: Forecast estimates "
        "demand; Erlang-C converts it into a staffing recommendation, and FAIR WORKFORCE "
        "COMPARISON evaluates that predefined schedule in the Dynamic Twin. Fixed staffing "
        "choices remain separate CAPACITY WHAT-IFS."
    )
    st.header("Workforce")
    st.caption(
        "Erlang-C analytical staffing recommendation · transparent M/M/c baseline, "
        "not a guaranteed operational outcome."
    )
    if not FORECAST_SERIES_PATH.is_file() or not FORECAST_ARTIFACT_PATH.is_file():
        st.error(
            "The Phase 9 forecast artifacts are required before workforce planning."
        )
        return
    _, forecast, _ = _load_forecast_view()
    _fair_workforce_comparison_panel(forecast)
    defaults = default_workforce_config()
    first, second, third = st.columns(3)
    target = float(
        first.slider(
            "Target service level",
            min_value=0.50,
            max_value=0.99,
            value=defaults.target_service_level,
            step=0.01,
        )
    )
    occupancy = float(
        second.slider(
            "Maximum occupancy",
            min_value=0.50,
            max_value=0.95,
            value=defaults.max_occupancy,
            step=0.01,
        )
    )
    buffer = float(
        third.select_slider(
            "Forecast safety buffer",
            options=(0, 5, 10, 15, 20, 25, 30),
            value=int(defaults.forecast_buffer_percent),
            format_func=lambda value: f"{value}%",
        )
    )
    fourth, fifth = st.columns(2)
    max_agents = int(
        fourth.number_input(
            "Maximum agents considered",
            min_value=1,
            max_value=200,
            value=defaults.max_agents,
            step=1,
        )
    )
    hold = int(
        fifth.number_input(
            "Intervals before reducing staffing",
            min_value=1,
            max_value=8,
            value=defaults.reduction_hold_intervals,
            step=1,
        )
    )
    st.caption(
        f"Calibrated mean AHT: {defaults.mean_aht_minutes:.3f} min · Service rate: "
        f"{defaults.service_rate_per_hour:.3f} contacts/hour/agent · "
        f"SLA threshold: {defaults.sla_wait_threshold_minutes:.1f} min. "
        "The 80% target and 85% occupancy defaults are configurable V1 planning assumptions."
    )
    if st.button("BUILD WORKFORCE PLAN", type="primary", width="stretch"):
        config_for_plan = default_workforce_config(
            target_service_level=target,
            max_occupancy=occupancy,
            forecast_buffer_percent=buffer,
            max_agents=max_agents,
            reduction_hold_intervals=hold,
        )
        st.session_state.manager_workforce_plan = build_workforce_plan(
            forecast, config_for_plan
        )

    plan = st.session_state.get("manager_workforce_plan")
    if plan is None:
        st.info("Choose explicit planning assumptions and press BUILD WORKFORCE PLAN.")
        _rl_experiment_panel()
        return
    summary = summarize_plan(plan)
    cards = st.columns(4)
    cards[0].metric("Peak planned agents", summary.maximum_agents)
    cards[1].metric("Average planned agents", f"{summary.average_agents:.2f}")
    cards[2].metric("Total agent-hours", f"{summary.total_agent_hours:.1f}")
    cards[3].metric(
        "Target-attainment slots",
        f"{summary.target_attainment_intervals} / {len(plan.points)}",
    )
    if summary.capacity_shortfall_intervals:
        st.error(
            f"Capacity shortfall in {summary.capacity_shortfall_intervals} intervals: "
            "configured maximum staffing cannot satisfy both analytical targets."
        )
    else:
        st.success("No analytical capacity shortfall under the selected assumptions.")

    rows = workforce_chart_rows(plan)
    st.subheader("Forecast demand")
    st.line_chart(
        rows, x="timestamp", y="forecast_contacts", x_label="Forecast timestamp"
    )
    st.subheader("Raw requirement vs operationalized staffing")
    st.line_chart(
        rows,
        x="timestamp",
        y=["raw_agents", "recommended_agents"],
        x_label="Forecast timestamp",
    )
    st.caption(
        "The operational schedule delays reductions for the configured number of intervals; "
        "raw Erlang-C requirements remain visible."
    )

    default_fixed = max(1, round(summary.average_agents))
    fixed_agents = int(
        st.number_input(
            "Fixed staffing baseline",
            min_value=1,
            max_value=200,
            value=default_fixed,
            step=1,
            key=f"fixed_baseline_{plan.forecast_origin.isoformat()}",
        )
    )
    comparison = compare_staffing_strategies(plan, fixed_agents)
    st.subheader("Analytical comparison · FIXED vs ERLANG-C")
    st.dataframe(
        [
            {
                "Strategy": item.strategy,
                "Definition": item.baseline_definition,
                "Agent-hours": item.total_agent_hours,
                "Average agents": item.average_agents,
                "Peak agents": item.peak_agents,
                "Target slots": f"{item.target_attainment_intervals} / 48",
                "Mean modeled utilization": f"{item.average_utilization:.1%}",
                "Mean modeled service level": f"{item.average_service_level:.1%}",
                "Shortfall slots": item.capacity_shortfall_intervals,
            }
            for item in (comparison.fixed, comparison.erlang_c)
        ],
        hide_index=True,
        width="stretch",
    )
    peak_point = max(plan.points, key=lambda point: point.recommended_agents)
    st.info("Peak-interval explanation: " + peak_point.explanation)

    if WORKFORCE_VALIDATION_PATH.is_file():
        validation = json.loads(WORKFORCE_VALIDATION_PATH.read_text(encoding="utf-8"))
        with st.expander("Digital Twin validation · simulated five-seed means"):
            st.warning(
                "Theoretical Erlang-C predictions and simulated Digital Twin outcomes are "
                "different model classes and are labelled separately."
            )
            st.dataframe(
                [
                    {
                        "Case": case["case"],
                        "Demand/hour": case["arrival_rate_per_hour"],
                        "Agents": case["agents"],
                        "Erlang utilization": case["erlang_c"]["utilization"],
                        "Twin utilization": case["digital_twin"]["mean_occupancy"],
                        "Erlang wait": case["erlang_c"]["expected_wait_minutes"],
                        "Twin wait": case["digital_twin"]["mean_average_wait_minutes"],
                        "Erlang SLA": case["erlang_c"]["service_level"],
                        "Twin SLA": case["digital_twin"]["mean_sla"],
                        "Twin abandonment": case["digital_twin"]["mean_abandonment"],
                    }
                    for case in validation["cases"]
                ],
                hide_index=True,
                width="stretch",
            )
    st.info(
        "The fixed 3→5 comparison remains a CAPACITY WHAT-IF. The fair comparison above "
        "is WORKFORCE INTELLIGENCE: fixed capacity versus a predefined 30-minute schedule "
        "under a lower total agent-hour budget."
    )
    _rl_experiment_panel()


def _rl_experiment_panel() -> None:
    st.divider()
    st.subheader("EXPERIMENTAL PPO POLICY — NOT ADOPTED")
    st.caption(
        "The learned policy reduced queue penalties through severe overstaffing in its simplified "
        "training environment, so it was not selected for operational use. The PPO training "
        "environment is not the calibrated Digital Twin."
    )
    if not rl_artifacts_available():
        st.info(
            "PPO experiment artifacts are unavailable. Train with "
            "`python -m callverse.rl.train --timesteps 50000`, then evaluate with "
            "`python -m callverse.rl.evaluation`."
        )
        return
    metadata = json.loads(RL_METADATA_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(RL_EVALUATION_PATH.read_text(encoding="utf-8"))
    strategies = evaluation["strategies"]
    rows = []
    for name in ("fixed", "erlang_c", "ppo"):
        metrics = strategies[name]["metrics"]
        rows.append(
            {
                "Strategy": name.upper(),
                "Definition": (
                    "learned experimental policy"
                    if name == "ppo"
                    else "analytical baseline"
                    if name == "erlang_c"
                    else f"constant {evaluation['fixed_agents']}-agent baseline"
                ),
                "Reward mean ± SD": (
                    f"{metrics['reward']['mean']:.2f} ± "
                    f"{metrics['reward']['standard_deviation']:.2f}"
                ),
                "Agent-hours/day": f"{metrics['total_agent_hours']['mean']:.1f}",
                "Average / peak agents": (
                    f"{metrics['average_staffing']['mean']:.1f} / "
                    f"{metrics['peak_staffing']['mean']:.1f}"
                ),
                "Abandonment": f"{metrics['abandonment_rate']['mean']:.2%}",
                "Service level": f"{metrics['service_level']['mean']:.2%}",
                "Average wait": f"{metrics['average_wait_minutes']['mean']:.2f} min",
                "Occupancy": f"{metrics['average_occupancy']['mean']:.2%}",
                "Staffing changes": f"{metrics['staffing_changes']['mean']:.1f}",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.error(
        "Operational recommendation: NOT ADOPTED. PPO achieved its higher environment reward "
        "by holding about "
        "17 agents (408 agent-hours/day), versus about 118 agent-hours for Erlang-C. "
        "This is overstaffing, not evidence that RL beats the analytical baseline."
    )
    trajectories = evaluation["representative_trajectory"]
    chart_rows = []
    for strategy, points in trajectories.items():
        for point in points:
            chart_rows.append(
                {"slot": point["slot"], "strategy": strategy, "agents": point["agents"]}
            )
    st.line_chart(chart_rows, x="slot", y="agents", color="strategy")
    with st.expander("Experiment reproducibility and limitations"):
        st.json(
            {
                "selected_seed": metadata["selected_seed"],
                "selection_rule": metadata["selection_rule"],
                "timesteps_per_seed": metadata["timesteps_per_seed"],
                "training_seeds": metadata["training_seeds"],
                "training_runtime_seconds": metadata["total_runtime_seconds"],
                "held_out_split": evaluation["held_out_split"],
                "evaluation_seeds": evaluation["stochastic_seeds"],
                "forecast_signal": evaluation["forecast_signal"],
            }
        )
        st.warning(
            "The compact RL environment batches arrivals into 30-minute decisions. Its fixed-"
            "staffing cross-check has similar occupancy direction but materially different wait, "
            "SLA, and abandonment from the calibrated continuous-time Digital Twin."
        )


def render_manager() -> None:
    st.title("CallVerse · Manager Control Room")
    st.write(
        "CallVerse helps a manager test operational decisions in a simulated support center "
        "before applying them."
    )
    st.caption(
        "Operational Digital Twin metrics and individual interaction quality are separate."
    )
    st.info(
        "**1 Scenario** → **2 Twin** → **3 Forecast** → **4 Workforce** → "
        "**5 Compare** → **6 Interaction** → **7 Quality**"
    )
    st.caption(
        "CallVerse V1 research decision-support prototype. This guided journey keeps the "
        "existing tabs available throughout."
    )
    with st.expander("Recommended demo path"):
        st.write(
            "Load Staff Shortage (seed 404, 3 agents, calibrated) → run the Digital Twin → "
            "inspect replay → review Forecast → run the fair Workforce comparison → optionally "
            "compare fixed 3→5 capacity → run an Interaction Lab case → review Quality. "
            "Briefly show PPO after Erlang-C."
        )
        st.caption("Each action remains explicit; this path does not auto-run any step.")
    tabs = st.tabs(
        [
            "Scenario Studio",
            "Twin Monitor",
            "Forecast",
            "Workforce",
            "Compare Decisions",
            "Interaction Lab",
            "Quality",
        ]
    )
    with tabs[0]:
        _scenario_studio()
    with tabs[1]:
        _twin_monitor()
    with tabs[2]:
        _forecast_panel()
    with tabs[3]:
        _workforce_panel()
    with tabs[4]:
        _compare_decisions()
    with tabs[5]:
        _interaction_lab()
    with tabs[6]:
        _quality_panel()
