"""Pure replay/view models over completed Digital Twin snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .scenario_guidance import get_scenario_guide
from .view_models import ManagerRun


class ReplayPressure(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass(frozen=True)
class ReplayContext:
    scenario_title: str
    risk_level: str
    manager_question: str
    available_agents: int
    seed: int
    policy_mode: str


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    frame_count: int
    simulation_minute: float
    simulated_clock: str
    queue_size: int
    busy_agents: int
    available_agents: int
    free_agents: int
    completed_count: int
    abandoned_count: int
    pressure: ReplayPressure
    pressure_fraction: float
    visible_waiting_icons: int
    hidden_waiting_count: int


def replay_context(run: ManagerRun) -> ReplayContext:
    guide = get_scenario_guide(run.scenario.name)
    return ReplayContext(
        scenario_title=guide.title,
        risk_level=guide.risk_level.value,
        manager_question=guide.manager_question,
        available_agents=run.result.available_agents,
        seed=run.result.seed,
        policy_mode=run.policy_mode,
    )


def simulated_clock(start_minute_of_day: int, elapsed_minutes: float) -> str:
    """Format elapsed simulation time as a clock without inventing a date."""

    minute = round((start_minute_of_day + elapsed_minutes) % 1440)
    hours, minutes = divmod(minute, 60)
    return f"{hours:02d}:{minutes:02d}"


def replay_pressure(
    queue_size: int, busy_agents: int, available_agents: int
) -> ReplayPressure:
    """Transparent descriptive pressure from queue and instantaneous staffing state."""

    if available_agents <= 0:
        raise ValueError("available agents must be positive")
    if queue_size >= available_agents * 2:
        return ReplayPressure.SEVERE
    if queue_size >= available_agents:
        return ReplayPressure.HIGH
    busy_ratio = busy_agents / available_agents
    if queue_size > 0 or busy_ratio >= 0.85:
        return ReplayPressure.MODERATE
    return ReplayPressure.LOW


def build_replay_frame(
    run: ManagerRun,
    frame_index: int,
    *,
    icon_cap: int = 10,
) -> ReplayFrame:
    """Select one existing snapshot; never invokes or mutates the simulator."""

    snapshots = run.result.snapshots
    if not snapshots:
        raise ValueError("simulation result has no snapshots")
    if not 0 <= frame_index < len(snapshots):
        raise IndexError("replay frame index is outside the snapshot range")
    if icon_cap < 0:
        raise ValueError("icon cap cannot be negative")

    snapshot = snapshots[frame_index]
    available = snapshot.available_agents
    free = max(available - snapshot.busy_agents, 0)
    pressure = replay_pressure(snapshot.queue_size, snapshot.busy_agents, available)
    pressure_fraction = {
        ReplayPressure.LOW: 0.20,
        ReplayPressure.MODERATE: 0.50,
        ReplayPressure.HIGH: 0.75,
        ReplayPressure.SEVERE: 1.00,
    }[pressure]
    visible_icons = min(snapshot.queue_size, icon_cap)
    return ReplayFrame(
        index=frame_index,
        frame_count=len(snapshots),
        simulation_minute=snapshot.simulation_time,
        simulated_clock=simulated_clock(
            run.scenario.simulation_start_minute_of_day,
            snapshot.simulation_time,
        ),
        queue_size=snapshot.queue_size,
        busy_agents=snapshot.busy_agents,
        available_agents=available,
        free_agents=free,
        completed_count=snapshot.completed_count,
        abandoned_count=snapshot.abandoned_count,
        pressure=pressure,
        pressure_fraction=pressure_fraction,
        visible_waiting_icons=visible_icons,
        hidden_waiting_count=snapshot.queue_size - visible_icons,
    )


def waiting_visual(frame: ReplayFrame) -> str:
    """Return a bounded representative queue display."""

    if frame.queue_size == 0:
        return "No contacts waiting"
    icons = " ".join("👤" for _ in range(frame.visible_waiting_icons))
    if frame.hidden_waiting_count:
        return f"{icons}  +{frame.hidden_waiting_count} more"
    return icons


def replay_slider_key(run: ManagerRun) -> str:
    """Scope replay state to the latest exact scenario configuration."""

    scenario = run.scenario
    return (
        f"twin_replay_{scenario.name}_{run.result.seed}_{scenario.available_agents}_"
        f"{scenario.demand_multiplier:g}_{scenario.simulation_duration:g}_{run.policy_mode}"
    )


def replay_option_label(run: ManagerRun, frame_index: int) -> str:
    frame = build_replay_frame(run, frame_index)
    return f"{frame.simulated_clock} · minute {frame.simulation_minute:g}"
