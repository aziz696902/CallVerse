"""Numerically stable Erlang-C staffing mathematics.

The formulation and staffing-sweep structure adapt the MIT-licensed
thelostbong/Queueing-Simulation-and-Optimization-System at revision
762d29ee4aac62d5e184ad8a2b5f524bed60dcd5. CallVerse replaces direct factorial
terms with the stable Erlang-B recurrence and adds service-level and occupancy
constraints.
"""

from __future__ import annotations

import math

from .models import ErlangCResult, StaffingSearchResult


def offered_load(arrival_rate_per_hour: float, service_rate_per_hour_per_agent: float) -> float:
    if arrival_rate_per_hour < 0:
        raise ValueError("arrival rate cannot be negative")
    if service_rate_per_hour_per_agent <= 0:
        raise ValueError("service rate must be positive")
    return arrival_rate_per_hour / service_rate_per_hour_per_agent


def utilization(
    arrival_rate_per_hour: float,
    service_rate_per_hour_per_agent: float,
    agents: int,
) -> float:
    load = offered_load(arrival_rate_per_hour, service_rate_per_hour_per_agent)
    if agents < 0:
        raise ValueError("agents cannot be negative")
    if agents == 0:
        return 0.0 if load == 0 else math.inf
    return load / agents


def _erlang_b(load: float, agents: int) -> float:
    """Stable recurrence avoiding powers and factorials."""
    blocking = 1.0
    for count in range(1, agents + 1):
        blocking = load * blocking / (count + load * blocking)
    return blocking


def evaluate_erlang_c(
    arrival_rate_per_hour: float,
    service_rate_per_hour_per_agent: float,
    agents: int,
    sla_wait_threshold_minutes: float,
) -> ErlangCResult:
    if agents < 0:
        raise ValueError("agents cannot be negative")
    if sla_wait_threshold_minutes < 0:
        raise ValueError("SLA wait threshold cannot be negative")
    load = offered_load(arrival_rate_per_hour, service_rate_per_hour_per_agent)
    rho = utilization(arrival_rate_per_hour, service_rate_per_hour_per_agent, agents)
    if arrival_rate_per_hour == 0:
        return ErlangCResult(
            arrival_rate_per_hour=0,
            service_rate_per_hour_per_agent=service_rate_per_hour_per_agent,
            agents=agents,
            offered_load=0,
            utilization=0,
            probability_wait=0,
            expected_wait_minutes=0,
            service_level=1,
            stable=True,
        )
    if agents == 0 or rho >= 1:
        return ErlangCResult(
            arrival_rate_per_hour=arrival_rate_per_hour,
            service_rate_per_hour_per_agent=service_rate_per_hour_per_agent,
            agents=agents,
            offered_load=load,
            utilization=rho,
            probability_wait=1,
            expected_wait_minutes=None,
            service_level=0,
            stable=False,
        )
    erlang_b = _erlang_b(load, agents)
    probability_wait = erlang_b / (1 - rho + rho * erlang_b)
    spare_capacity_per_hour = agents * service_rate_per_hour_per_agent - arrival_rate_per_hour
    expected_wait_minutes = probability_wait / spare_capacity_per_hour * 60
    threshold_hours = sla_wait_threshold_minutes / 60
    service_level = 1 - probability_wait * math.exp(-spare_capacity_per_hour * threshold_hours)
    return ErlangCResult(
        arrival_rate_per_hour=arrival_rate_per_hour,
        service_rate_per_hour_per_agent=service_rate_per_hour_per_agent,
        agents=agents,
        offered_load=load,
        utilization=rho,
        probability_wait=min(1.0, max(0.0, probability_wait)),
        expected_wait_minutes=max(0.0, expected_wait_minutes),
        service_level=min(1.0, max(0.0, service_level)),
        stable=True,
    )


def minimum_agents(
    *,
    arrival_rate_per_hour: float,
    service_rate_per_hour_per_agent: float,
    sla_wait_threshold_minutes: float,
    target_service_level: float,
    max_occupancy: float,
    min_agents: int,
    max_agents: int,
) -> StaffingSearchResult:
    if not 0 <= target_service_level <= 1:
        raise ValueError("target service level must be between zero and one")
    if not 0 < max_occupancy < 1:
        raise ValueError("maximum occupancy must be between zero and one")
    if min_agents < 0 or max_agents < min_agents:
        raise ValueError("invalid staffing search range")
    for agents in range(min_agents, max_agents + 1):
        metrics = evaluate_erlang_c(
            arrival_rate_per_hour,
            service_rate_per_hour_per_agent,
            agents,
            sla_wait_threshold_minutes,
        )
        if (
            metrics.stable
            and metrics.service_level >= target_service_level
            and metrics.utilization <= max_occupancy
        ):
            return StaffingSearchResult(
                agents=agents,
                metrics=metrics,
                target_met=True,
                capacity_shortfall=False,
            )
    metrics = evaluate_erlang_c(
        arrival_rate_per_hour,
        service_rate_per_hour_per_agent,
        max_agents,
        sla_wait_threshold_minutes,
    )
    return StaffingSearchResult(
        agents=max_agents,
        metrics=metrics,
        target_met=False,
        capacity_shortfall=True,
    )

