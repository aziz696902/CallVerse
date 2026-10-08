"""Pure synchronized replay models over two completed Digital Twin runs."""

from __future__ import annotations

from enum import Enum
from itertools import pairwise
from math import isclose

from pydantic import Field

from callverse.domain import DomainModel
from callverse.scenarios import get_scenario

from .twin_replay import ReplayPressure, build_replay_frame
from .view_models import (
    DecisionComparison,
    ManagerRun,
    PolicyMode,
    configure_scenario,
    run_manager_simulation,
    run_staffing_what_if,
)

BASELINE_LABEL = "FIXED STAFFING BASELINE"
ASSISTED_LABEL = "CALLVERSE-ASSISTED DECISION"


class DecisionSource(str, Enum):
    MANAGER_SELECTED = "Manager-selected decision"
    CAPACITY_WHAT_IF = "Capacity What-if"
    SAME_STAFF_CONTROL = "Same-staff control"
    WORKFORCE_RECOMMENDATION = "Workforce recommendation"
    SCALABILITY_DEMO = "Scalability demonstration"


class ComparisonConfiguration(DomainModel):
    scenario_name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    duration_minutes: float = Field(gt=0)
    demand_multiplier: float = Field(gt=0)
    baseline_agents: int = Field(ge=1)
    assisted_agents: int = Field(ge=1)
    policy_mode: PolicyMode
    decision_source: DecisionSource = DecisionSource.MANAGER_SELECTED


class ReplaySideFrame(DomainModel):
    label: str
    available_agents: int = Field(ge=1)
    queue_size: int = Field(ge=0)
    busy_agents: int = Field(ge=0)
    free_agents: int = Field(ge=0)
    completed_so_far: int = Field(ge=0)
    abandoned_so_far: int = Field(ge=0)
    snapshot_pressure: ReplayPressure


class ComparativeFrame(DomainModel):
    index: int = Field(ge=0)
    frame_count: int = Field(ge=1)
    simulation_minute: float = Field(ge=0)
    simulated_clock: str
    baseline: ReplaySideFrame
    assisted: ReplaySideFrame


class ComparativeReplay(DomainModel):
    configuration: ComparisonConfiguration
    comparison: DecisionComparison
    frames: tuple[ComparativeFrame, ...]
    snapshot_cadence_minutes: float | None = Field(default=None, gt=0)
    changed_parameter: str
    held_constant: tuple[str, ...]


def create_comparison_configuration(
    scenario_name: str,
    *,
    seed: int,
    duration_minutes: float,
    demand_multiplier: float,
    baseline_agents: int,
    assisted_agents: int,
    policy_mode: PolicyMode,
    decision_source: DecisionSource = DecisionSource.MANAGER_SELECTED,
) -> ComparisonConfiguration:
    """Validate a comparison setup without running either simulation."""

    get_scenario(scenario_name)
    return ComparisonConfiguration(
        scenario_name=scenario_name,
        seed=seed,
        duration_minutes=duration_minutes,
        demand_multiplier=demand_multiplier,
        baseline_agents=baseline_agents,
        assisted_agents=assisted_agents,
        policy_mode=policy_mode,
        decision_source=decision_source,
    )


def capacity_what_if_configuration() -> ComparisonConfiguration:
    """Return the secondary jury configuration that adds fixed capacity."""

    scenario = get_scenario("staff_shortage")
    return create_comparison_configuration(
        scenario.name,
        seed=404,
        duration_minutes=scenario.simulation_duration,
        demand_multiplier=scenario.demand_multiplier,
        baseline_agents=3,
        assisted_agents=5,
        policy_mode="calibrated",
        decision_source=DecisionSource.CAPACITY_WHAT_IF,
    )


def same_staff_control_configuration() -> ComparisonConfiguration:
    """Return the scientific control with identical fixed staffing."""

    scenario = get_scenario("staff_shortage")
    return create_comparison_configuration(
        scenario.name,
        seed=404,
        duration_minutes=scenario.simulation_duration,
        demand_multiplier=scenario.demand_multiplier,
        baseline_agents=3,
        assisted_agents=3,
        policy_mode="calibrated",
        decision_source=DecisionSource.SAME_STAFF_CONTROL,
    )


def _cadence(timestamps: tuple[float, ...]) -> float | None:
    if len(timestamps) < 2:
        return None
    deltas = tuple(
        current - previous
        for previous, current in pairwise(timestamps)
    )
    first = deltas[0]
    if first <= 0 or not all(isclose(delta, first, abs_tol=1e-9) for delta in deltas):
        return None
    return first


def _side_frame(run: ManagerRun, index: int, label: str) -> ReplaySideFrame:
    replay = build_replay_frame(run, index)
    return ReplaySideFrame(
        label=label,
        available_agents=replay.available_agents,
        queue_size=replay.queue_size,
        busy_agents=replay.busy_agents,
        free_agents=replay.free_agents,
        completed_so_far=replay.completed_count,
        abandoned_so_far=replay.abandoned_count,
        snapshot_pressure=replay.pressure,
    )


def _aligned_frames(comparison: DecisionComparison) -> tuple[ComparativeFrame, ...]:
    baseline_run = comparison.before
    assisted_run = comparison.after
    baseline_times = tuple(
        snapshot.simulation_time for snapshot in baseline_run.result.snapshots
    )
    assisted_times = tuple(
        snapshot.simulation_time for snapshot in assisted_run.result.snapshots
    )
    if not baseline_times or not assisted_times:
        raise ValueError("both comparison runs must contain replay snapshots")
    if baseline_times != assisted_times:
        raise ValueError("comparison snapshot timestamps do not align exactly")

    frame_count = len(baseline_times)
    frames = []
    for index, simulation_minute in enumerate(baseline_times):
        baseline = build_replay_frame(baseline_run, index)
        assisted = build_replay_frame(assisted_run, index)
        if baseline.simulated_clock != assisted.simulated_clock:
            raise ValueError("comparison snapshot clock labels do not align exactly")
        frames.append(
            ComparativeFrame(
                index=index,
                frame_count=frame_count,
                simulation_minute=simulation_minute,
                simulated_clock=baseline.simulated_clock,
                baseline=_side_frame(baseline_run, index, BASELINE_LABEL),
                assisted=_side_frame(assisted_run, index, ASSISTED_LABEL),
            )
        )
    return tuple(frames)


def build_comparative_replay(
    configuration: ComparisonConfiguration,
) -> ComparativeReplay:
    """Run each existing Twin path once, then build immutable synchronized frames."""

    baseline_scenario = configure_scenario(
        configuration.scenario_name,
        seed=configuration.seed,
        available_agents=configuration.baseline_agents,
        demand_multiplier=configuration.demand_multiplier,
        duration=configuration.duration_minutes,
    )
    baseline_run = run_manager_simulation(
        baseline_scenario, configuration.policy_mode
    )
    comparison = run_staffing_what_if(
        baseline_run, configuration.assisted_agents
    )

    baseline_fields = baseline_run.scenario.model_dump()
    assisted_fields = comparison.after.scenario.model_dump()
    baseline_fields.pop("available_agents")
    assisted_fields.pop("available_agents")
    if baseline_fields != assisted_fields:
        raise ValueError("comparison changed a scenario field other than staffing")

    frames = _aligned_frames(comparison)
    timestamps = tuple(frame.simulation_minute for frame in frames)
    return ComparativeReplay(
        configuration=configuration,
        comparison=comparison,
        frames=frames,
        snapshot_cadence_minutes=_cadence(timestamps),
        changed_parameter=(
            f"Available advisors: {configuration.baseline_agents} → "
            f"{configuration.assisted_agents}"
        ),
        held_constant=(
            f"Scenario: {configuration.scenario_name}",
            f"Seed: {configuration.seed}",
            f"Duration: {configuration.duration_minutes:g} simulated minutes",
            f"Demand multiplier: {configuration.demand_multiplier:g}x",
            f"Simulator mode: {configuration.policy_mode}",
            "Intent mix, persona mix, and all other scenario settings",
        ),
    )


def select_comparative_frame(
    replay: ComparativeReplay, frame_index: int
) -> ComparativeFrame:
    """Select an already-built frame without invoking either simulation."""

    if not 0 <= frame_index < len(replay.frames):
        raise IndexError("comparative replay frame index is outside the timeline")
    return replay.frames[frame_index]


def comparison_matches_configuration(
    replay: ComparativeReplay, configuration: ComparisonConfiguration
) -> bool:
    """Return whether stored replay data belongs to the visible setup controls."""

    return replay.configuration == configuration


def comparative_slider_key(configuration: ComparisonConfiguration) -> str:
    """Scope manual timeline state to one exact comparison configuration."""

    return (
        f"comparative_replay_{configuration.scenario_name}_{configuration.seed}_"
        f"{configuration.baseline_agents}_{configuration.assisted_agents}_"
        f"{configuration.demand_multiplier:g}_{configuration.duration_minutes:g}_"
        f"{configuration.policy_mode}_{configuration.decision_source.value}"
    )
