"""Deterministic storytelling over immutable comparative replay frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .comparative_replay import (
    ASSISTED_LABEL,
    BASELINE_LABEL,
    ComparativeFrame,
    ComparativeReplay,
    ComparisonConfiguration,
    DecisionSource,
    ReplaySideFrame,
    create_comparison_configuration,
)
from .decision_guidance import (
    ChangeAssessment,
    DecisionNarrative,
    build_decision_narrative,
)

LARGE_CENTER_SCALE = 5
LARGE_CENTER_DEMAND_MULTIPLIER = 1.15 * LARGE_CENTER_SCALE
LARGE_CENTER_BASELINE_AGENTS = 3 * LARGE_CENTER_SCALE
LARGE_CENTER_ASSISTED_AGENTS = 5 * LARGE_CENTER_SCALE


class ReplayEventType(str, Enum):
    SIMULATION_START = "SIMULATION_START"
    QUEUE_EMERGES = "QUEUE_EMERGES"
    QUEUE_CLEARS = "QUEUE_CLEARS"
    ALL_ADVISORS_BUSY = "ALL_ADVISORS_BUSY"
    PRESSURE_MODERATE = "PRESSURE_MODERATE"
    PRESSURE_HIGH = "PRESSURE_HIGH"
    PRESSURE_SEVERE = "PRESSURE_SEVERE"
    SIMULATION_END = "SIMULATION_END"


@dataclass(frozen=True)
class ReplayEvent:
    frame_index: int
    simulation_time: float
    simulated_clock: str
    side: Literal["comparison", "baseline", "assisted"]
    event_type: ReplayEventType
    title: str
    description: str


@dataclass(frozen=True)
class FrameDifference:
    metric: str
    baseline: int
    assisted: int
    assisted_delta: int
    semantics: Literal["current snapshot", "cumulative from simulation start"]


@dataclass(frozen=True)
class FinalManagerStory:
    narrative: DecisionNarrative
    staffing_delta: int
    staffing_trade_off: str
    service_effects: tuple[str, ...]
    scientific_boundary: str


def large_center_configuration() -> ComparisonConfiguration:
    """Return the predefined 5x scalability demonstration configuration."""

    return create_comparison_configuration(
        "staff_shortage",
        seed=404,
        duration_minutes=480,
        demand_multiplier=LARGE_CENTER_DEMAND_MULTIPLIER,
        baseline_agents=LARGE_CENTER_BASELINE_AGENTS,
        assisted_agents=LARGE_CENTER_ASSISTED_AGENTS,
        policy_mode="calibrated",
        decision_source=DecisionSource.SCALABILITY_DEMO,
    )


def _first_index(
    sides: tuple[ReplaySideFrame, ...], predicate: Callable[[ReplaySideFrame], bool]
) -> int | None:
    return next((index for index, side in enumerate(sides) if predicate(side)), None)


def _event(
    replay: ComparativeReplay,
    index: int,
    side: Literal["comparison", "baseline", "assisted"],
    event_type: ReplayEventType,
    title: str,
    description: str,
) -> ReplayEvent:
    frame = replay.frames[index]
    return ReplayEvent(
        frame_index=index,
        simulation_time=frame.simulation_minute,
        simulated_clock=frame.simulated_clock,
        side=side,
        event_type=event_type,
        title=title,
        description=description,
    )


def generate_replay_events(replay: ComparativeReplay) -> tuple[ReplayEvent, ...]:
    """Generate first-occurrence markers supported directly by stored frame values."""

    events = [
        _event(
            replay,
            0,
            "comparison",
            ReplayEventType.SIMULATION_START,
            "Simulation begins",
            (
                f"Baseline starts with {replay.configuration.baseline_agents} advisors; "
                f"assisted starts with {replay.configuration.assisted_agents} advisors."
            ),
        )
    ]
    for side_name, label in (
        ("baseline", "Baseline"),
        ("assisted", "Assisted"),
    ):
        sides = tuple(getattr(frame, side_name) for frame in replay.frames)
        queue_emerges = _first_index(sides, lambda side: side.queue_size > 0)
        if queue_emerges is not None:
            events.append(
                _event(
                    replay,
                    queue_emerges,
                    side_name,
                    ReplayEventType.QUEUE_EMERGES,
                    f"{label}: queue emerges",
                    "The stored snapshot first shows waiting contacts.",
                )
            )
        queue_clears = next(
            (
                index
                for index in range(1, len(sides))
                if sides[index - 1].queue_size > 0 and sides[index].queue_size == 0
            ),
            None,
        )
        if queue_clears is not None:
            events.append(
                _event(
                    replay,
                    queue_clears,
                    side_name,
                    ReplayEventType.QUEUE_CLEARS,
                    f"{label}: sampled queue clears",
                    "The current stored snapshot shows no waiting contacts.",
                )
            )
        all_busy = _first_index(
            sides, lambda side: side.busy_agents == side.available_agents
        )
        if all_busy is not None:
            events.append(
                _event(
                    replay,
                    all_busy,
                    side_name,
                    ReplayEventType.ALL_ADVISORS_BUSY,
                    f"{label}: all advisors busy",
                    "Busy advisors equal available advisors at this snapshot.",
                )
            )
        for pressure, event_type in (
            ("MODERATE", ReplayEventType.PRESSURE_MODERATE),
            ("HIGH", ReplayEventType.PRESSURE_HIGH),
            ("SEVERE", ReplayEventType.PRESSURE_SEVERE),
        ):
            pressure_index = _first_index(
                sides, lambda side, target=pressure: side.snapshot_pressure.value == target
            )
            if pressure_index is not None:
                events.append(
                    _event(
                        replay,
                        pressure_index,
                        side_name,
                        event_type,
                        f"{label}: {pressure.lower()} snapshot pressure",
                        f"Current snapshot pressure first reaches {pressure}.",
                    )
                )
    events.append(
        _event(
            replay,
            len(replay.frames) - 1,
            "comparison",
            ReplayEventType.SIMULATION_END,
            "Simulation ends",
            "Both completed runs reach the shared simulation horizon.",
        )
    )
    side_order = {"comparison": 0, "baseline": 1, "assisted": 2}
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.frame_index,
                side_order[event.side],
                event.event_type.value,
            ),
        )
    )


def events_at_frame(
    events: tuple[ReplayEvent, ...], frame_index: int
) -> tuple[ReplayEvent, ...]:
    return tuple(event for event in events if event.frame_index == frame_index)


def current_frame_differences(
    frame: ComparativeFrame,
) -> tuple[FrameDifference, ...]:
    return (
        FrameDifference(
            metric="Current queue",
            baseline=frame.baseline.queue_size,
            assisted=frame.assisted.queue_size,
            assisted_delta=frame.assisted.queue_size - frame.baseline.queue_size,
            semantics="current snapshot",
        ),
        FrameDifference(
            metric="Completed so far",
            baseline=frame.baseline.completed_so_far,
            assisted=frame.assisted.completed_so_far,
            assisted_delta=(
                frame.assisted.completed_so_far - frame.baseline.completed_so_far
            ),
            semantics="cumulative from simulation start",
        ),
        FrameDifference(
            metric="Abandoned so far",
            baseline=frame.baseline.abandoned_so_far,
            assisted=frame.assisted.abandoned_so_far,
            assisted_delta=(
                frame.assisted.abandoned_so_far - frame.baseline.abandoned_so_far
            ),
            semantics="cumulative from simulation start",
        ),
        FrameDifference(
            metric="Busy advisors",
            baseline=frame.baseline.busy_agents,
            assisted=frame.assisted.busy_agents,
            assisted_delta=frame.assisted.busy_agents - frame.baseline.busy_agents,
            semantics="current snapshot",
        ),
    )


def queue_trajectory_rows(replay: ComparativeReplay) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "simulation_time": frame.simulation_minute,
            BASELINE_LABEL: frame.baseline.queue_size,
            ASSISTED_LABEL: frame.assisted.queue_size,
        }
        for frame in replay.frames
    )


def build_final_manager_story(replay: ComparativeReplay) -> FinalManagerStory:
    """Reuse decision guidance and add only deterministic presentation wording."""

    narrative = build_decision_narrative(replay.comparison)
    changes = {change.key: change for change in narrative.changes}
    effect_labels = {
        "sla": "Improved SLA",
        "abandonment": "Lower abandonment",
        "average_wait": "Lower average wait",
        "completed": "Higher completed contacts",
        "final_backlog": "Lower final backlog",
    }
    effects = tuple(
        effect_labels[key]
        for key in effect_labels
        if key in changes and changes[key].assessment is ChangeAssessment.BETTER
    )
    staffing_delta = (
        replay.configuration.assisted_agents - replay.configuration.baseline_agents
    )
    if staffing_delta > 0:
        staffing_trade_off = f"Additional staffing: +{staffing_delta} advisors"
    elif staffing_delta < 0:
        staffing_trade_off = f"Reduced staffing: {staffing_delta} advisors"
    else:
        staffing_trade_off = "Staffing difference: 0 advisors"
    return FinalManagerStory(
        narrative=narrative,
        staffing_delta=staffing_delta,
        staffing_trade_off=staffing_trade_off,
        service_effects=effects,
        scientific_boundary=(
            "This result reflects the matched simulated scenario and does not establish "
            "a production optimum."
        ),
    )
