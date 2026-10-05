"""Translate compact empirical support profiles into simulator policies."""

from __future__ import annotations

from callverse.domain import CustomerPersona, RequestIntent
from callverse.simulation.policies import (
    DEFAULT_POLICY,
    PROTOTYPE_HANDLING_TIMES,
    PROTOTYPE_PATIENCE,
    QuantileMinutes,
    ScaledMinutes,
    SimulationPolicy,
)

from .profiles import SupportCalibrationProfile


def build_calibrated_policy(profile: SupportCalibrationProfile) -> SimulationPolicy:
    """Use empirical shapes while retaining explicit provisional domain modifiers."""

    service = QuantileMinutes(profile.service_time_quantiles.probabilities, profile.service_time_quantiles.values)
    patience = QuantileMinutes(profile.patience_quantiles.probabilities, profile.patience_quantiles.values)
    general_mode = PROTOTYPE_HANDLING_TIMES[RequestIntent.GENERAL].mode
    new_mode = PROTOTYPE_PATIENCE[CustomerPersona.NEW].mode
    return SimulationPolicy(
        base_arrival_rate_per_minute=DEFAULT_POLICY.base_arrival_rate_per_minute,
        sla_target_minutes=DEFAULT_POLICY.sla_target_minutes,
        snapshot_interval_minutes=DEFAULT_POLICY.snapshot_interval_minutes,
        max_event_records=DEFAULT_POLICY.max_event_records,
        handling_times={intent: ScaledMinutes(service, values.mode / general_mode) for intent, values in PROTOTYPE_HANDLING_TIMES.items()},
        patience_times={persona: ScaledMinutes(patience, values.mode / new_mode) for persona, values in PROTOTYPE_PATIENCE.items()},
        arrival_slot_multipliers=tuple(slot.multiplier for slot in profile.arrival_slots),
    )
