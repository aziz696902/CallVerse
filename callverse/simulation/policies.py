"""Centralized prototype assumptions for Digital Twin V1.

These values are deliberately explicit and are not research-calibrated. Later work
can replace them with distributions estimated from suitable operational datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

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


class MinutesSampler(Protocol):
    def sample(self, rng: Random) -> float: ...


@dataclass(frozen=True)
class QuantileMinutes:
    """Piecewise-linear inverse CDF represented by compact empirical quantiles."""

    probabilities: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(self.values) or len(self.values) < 2:
            raise ValueError("quantile probabilities and values must have equal length >= 2")
        if self.probabilities[0] != 0 or self.probabilities[-1] != 1:
            raise ValueError("quantile probabilities must span 0 through 1")
        if any(a >= b for a, b in zip(self.probabilities, self.probabilities[1:])):
            raise ValueError("quantile probabilities must be strictly increasing")
        if any(value < 0 for value in self.values):
            raise ValueError("quantile durations must be non-negative")
        if any(a > b for a, b in zip(self.values, self.values[1:])):
            raise ValueError("quantile durations must be nondecreasing")

    def sample(self, rng: Random) -> float:
        probability = rng.random()
        for index in range(1, len(self.probabilities)):
            upper_probability = self.probabilities[index]
            if probability <= upper_probability:
                lower_probability = self.probabilities[index - 1]
                fraction = (probability - lower_probability) / (
                    upper_probability - lower_probability
                )
                lower_value = self.values[index - 1]
                upper_value = self.values[index]
                return lower_value + fraction * (upper_value - lower_value)
        return self.values[-1]


@dataclass(frozen=True)
class ScaledMinutes:
    base: MinutesSampler
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("duration scale must be positive")

    def sample(self, rng: Random) -> float:
        return self.base.sample(rng) * self.scale


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
    handling_times: dict[RequestIntent, MinutesSampler] = field(
        default_factory=lambda: dict(PROTOTYPE_HANDLING_TIMES)
    )
    patience_times: dict[CustomerPersona, MinutesSampler] = field(
        default_factory=lambda: dict(PROTOTYPE_PATIENCE)
    )
    arrival_slot_multipliers: tuple[float, ...] | None = None

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
        if self.arrival_slot_multipliers is not None:
            if len(self.arrival_slot_multipliers) != 48:
                raise ValueError("arrival profile must contain 48 half-hour multipliers")
            if any(multiplier < 0 for multiplier in self.arrival_slot_multipliers):
                raise ValueError("arrival multipliers must be non-negative")
            if max(self.arrival_slot_multipliers, default=0) <= 0:
                raise ValueError("arrival profile must contain a positive multiplier")

    def arrival_multiplier(self, simulation_minute: float, start_minute_of_day: int) -> float:
        if self.arrival_slot_multipliers is None:
            return 1.0
        minute_of_day = (start_minute_of_day + simulation_minute) % 1440
        return self.arrival_slot_multipliers[int(minute_of_day // 30)]

    def mean_arrival_multiplier(self, duration: float, start_minute_of_day: int) -> float:
        """Return the time-weighted mean profile weight over a scenario horizon."""

        if duration <= 0:
            raise ValueError("simulation duration must be positive")
        if self.arrival_slot_multipliers is None:
            return 1.0
        elapsed = 0.0
        weighted_sum = 0.0
        while elapsed < duration:
            minute_of_day = (start_minute_of_day + elapsed) % 1440
            until_boundary = 30 - minute_of_day % 30
            segment = min(until_boundary, duration - elapsed)
            weighted_sum += self.arrival_multiplier(elapsed, start_minute_of_day) * segment
            elapsed += segment
        mean = weighted_sum / duration
        if mean <= 0:
            raise ValueError("arrival profile has zero weight across the scenario horizon")
        return mean

    def normalized_arrival_multiplier(
        self,
        simulation_minute: float,
        start_minute_of_day: int,
        duration: float,
    ) -> float:
        """Weight time of day while preserving baseline expected horizon demand."""

        return self.arrival_multiplier(
            simulation_minute, start_minute_of_day
        ) / self.mean_arrival_multiplier(duration, start_minute_of_day)


DEFAULT_POLICY = SimulationPolicy()
