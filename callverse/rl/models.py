"""Small, serializable contracts for the experimental workforce policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardWeights:
    service_level: float = 2.0
    waiting: float = 0.35
    abandonment: float = 6.0
    staffing: float = 0.20
    overload: float = 1.5
    staffing_change: float = 0.08

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentConfig:
    horizon: int = 48
    interval_minutes: int = 30
    min_agents: int = 1
    max_agents: int = 20
    target_service_level: float = 0.80
    max_occupancy: float = 0.85
    sla_minutes: float = 2.0
    demand_scale: float = 120.0
    backlog_scale: float = 120.0


@dataclass(frozen=True)
class EpisodeMetrics:
    reward: float
    total_contacts: int
    completed: int
    abandoned: int
    remaining: int
    abandonment_rate: float
    service_level: float
    average_wait_minutes: float
    average_occupancy: float
    total_agent_hours: float
    staffing_changes: int
