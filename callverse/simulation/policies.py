"""Centralized prototype assumptions for Digital Twin V1.

These values are deliberately explicit and are not research-calibrated. Later work
can replace them with distributions estimated from suitable operational datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from callverse.domain import CustomerPersona, RequestIntent


@dataclass(frozen=True)
class TriangularMinutes:
    minimum: float
    mode: float
    maximum: float

    def __post_init__(self) -> None:
        if not 0 < self.minimum <= self.mode <= self.maximum:
            raise ValueError("triangular minutes require 0 < minimum <= mode <= maximum")

    def sample(self, rng: Random) -> float:
        return rng.triangular(self.minimum, self.maximum, self.mode)


PROTOTYPE_HANDLING_TIMES: dict[RequestIntent, TriangularMinutes] = {
    RequestIntent.TRACKING: TriangularMinutes(2.0, 4.0, 7.0),
    RequestIntent.REFUND: TriangularMinutes(5.0, 8.0, 13.0),
    RequestIntent.DAMAGED_ITEM: TriangularMinutes(7.0, 11.0, 17.0),
    RequestIntent.ADDRESS_CHANGE: TriangularMinutes(3.0, 5.0, 8.0),
    RequestIntent.CANCEL_ORDER: TriangularMinutes(4.0, 6.0, 10.0),
    RequestIntent.PAYMENT_ISSUE: TriangularMinutes(5.0, 8.0, 14.0),
    RequestIntent.COMPLAINT: TriangularMinutes(7.0, 12.0, 18.0),
    RequestIntent.GENERAL: TriangularMinutes(3.0, 6.0, 10.0),
}


PROTOTYPE_PATIENCE: dict[CustomerPersona, TriangularMinutes] = {
    CustomerPersona.NEW: TriangularMinutes(4.0, 8.0, 14.0),
    CustomerPersona.LOYAL: TriangularMinutes(7.0, 12.0, 20.0),
    CustomerPersona.UNHAPPY: TriangularMinutes(2.0, 4.0, 8.0),
    CustomerPersona.PREMIUM: TriangularMinutes(5.0, 9.0, 15.0),
    CustomerPersona.AT_RISK: TriangularMinutes(1.5, 3.5, 7.0),
}


@dataclass(frozen=True)
class SimulationPolicy:
    """Calibratable operational assumptions kept outside the simulation engine."""

    base_arrival_rate_per_minute: float = 0.75
    sla_target_minutes: float = 2.0
    snapshot_interval_minutes: float = 15.0
    max_event_records: int = 2_000
    handling_times: dict[RequestIntent, TriangularMinutes] = field(
        default_factory=lambda: dict(PROTOTYPE_HANDLING_TIMES)
    )
    patience_times: dict[CustomerPersona, TriangularMinutes] = field(
        default_factory=lambda: dict(PROTOTYPE_PATIENCE)
    )

    def __post_init__(self) -> None:
        if self.base_arrival_rate_per_minute <= 0:
            raise ValueError("base arrival rate must be positive")
        if self.sla_target_minutes < 0:
            raise ValueError("SLA target must be non-negative")
        if self.snapshot_interval_minutes <= 0:
            raise ValueError("snapshot interval must be positive")
        if self.max_event_records < 0:
            raise ValueError("maximum event-record count must be non-negative")
        if set(self.handling_times) != set(RequestIntent):
            raise ValueError("handling-time policy must cover every request intent")
        if set(self.patience_times) != set(CustomerPersona):
            raise ValueError("patience policy must cover every customer persona")


DEFAULT_POLICY = SimulationPolicy()
