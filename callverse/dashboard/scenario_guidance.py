"""Business-first guidance for the Manager Scenario Studio.

This module is intentionally separate from :mod:`callverse.scenarios`: its labels,
targets, and explanations are presentation metadata, not simulation inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal

from callverse.simulation import SimulationResult


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    SEVERE = "Severe"


class CenterStatus(str, Enum):
    HEALTHY = "HEALTHY"
    UNDER_PRESSURE = "UNDER PRESSURE"
    OVERLOADED = "OVERLOADED"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ScenarioGuide:
    scenario_key: str
    title: str
    short_description: str
    situation: str
    manager_question: str
    expected_behavior: str
    objectives: tuple[str, ...]
    suggested_action: str
    risk_level: RiskLevel
    demo_recommended: bool = False
    scientific_note: str = ""


@dataclass(frozen=True)
class ManagerTargets:
    """Expert-defined V1 interpretation targets, not learned thresholds."""

    minimum_sla: float = 0.80
    maximum_abandonment: float = 0.10
    maximum_occupancy: float = 0.85


@dataclass(frozen=True)
class TargetCheck:
    metric: str
    objective: str
    actual: float | None
    outcome: Literal["pass", "fail", "warning", "unavailable"]


@dataclass(frozen=True)
class ResultInterpretation:
    status: CenterStatus
    reasons: tuple[str, ...]
    target_checks: tuple[TargetCheck, ...]
    next_step: str


@dataclass(frozen=True)
class RecommendedDemo:
    scenario_key: str = "staff_shortage"
    seed: int = 404
    available_agents: int = 3
    policy_mode: str = "calibrated"


DEFAULT_TARGETS = ManagerTargets()
RECOMMENDED_DEMO = RecommendedDemo()
OBJECTIVES = (
    "SLA at least 80%",
    "Abandonment below 10%",
    "Occupancy below 85%",
)


_GUIDES = {
    "normal_day": ScenarioGuide(
        scenario_key="normal_day",
        title="Normal Day",
        short_description="A balanced baseline for understanding ordinary center flow.",
        situation="Baseline demand, eight advisors, and balanced customer and request mixes.",
        manager_question="Does current capacity comfortably meet the V1 operating targets?",
        expected_behavior="Designed to provide a low-pressure reference run for later comparisons.",
        objectives=OBJECTIVES,
        suggested_action="Run the baseline, then open Forecast to inspect the next 24 hours.",
        risk_level=RiskLevel.LOW,
        scientific_note="This is a simulated baseline, not a production observation.",
    ),
    "rainy_peak": ScenarioGuide(
        scenario_key="rainy_peak",
        title="Rainy Peak",
        short_description="Elevated demand with a tracking-heavy request mix.",
        situation="Demand is set to 1.35x with eight advisors and more tracking contacts.",
        manager_question="Can existing staffing absorb a busier, tracking-heavy period?",
        expected_behavior="Designed to test moderate pressure without reducing baseline staffing.",
        objectives=OBJECTIVES,
        suggested_action="Run the Twin and inspect queue and occupancy in Twin Monitor.",
        risk_level=RiskLevel.MODERATE,
        scientific_note=(
            "Rain is context only. The configured demand and request mix affect the Twin; "
            "no validated causal weather effect is modeled."
        ),
    ),
    "flash_sale": ScenarioGuide(
        scenario_key="flash_sale",
        title="Flash Sale",
        short_description="A short, high-volume promotion with a broad request mix.",
        situation="Demand is set to 2.20x for six hours with ten advisors and more new customers.",
        manager_question="Is added staffing sufficient for the simulated promotion load?",
        expected_behavior="Designed to test high throughput and possible queue pressure.",
        objectives=OBJECTIVES,
        suggested_action="Run the Twin, then test a staffing alternative if targets are missed.",
        risk_level=RiskLevel.HIGH,
        scientific_note="The promotion label describes this configured simulation only.",
    ),
    "staff_shortage": ScenarioGuide(
        scenario_key="staff_shortage",
        title="Staff Shortage",
        short_description="Moderately elevated demand with only three available advisors.",
        situation="Demand is set to 1.15x while staffing falls to three advisors.",
        manager_question="How much operational pressure does constrained staffing create?",
        expected_behavior="Designed to expose capacity pressure and support a same-seed staffing test.",
        objectives=OBJECTIVES,
        suggested_action="Observe the run, then use Compare Decisions to test more advisors.",
        risk_level=RiskLevel.HIGH,
        demo_recommended=True,
        scientific_note=(
            "Jury starting scenario; primary intelligence evidence comes from the separate "
            "lower-budget Dynamic Workforce comparison."
        ),
    ),
    "customer_crisis": ScenarioGuide(
        scenario_key="customer_crisis",
        title="Customer Crisis",
        short_description="A complaint-heavy, refund-heavy mix with less patient customers.",
        situation=(
            "Demand is set to 1.30x with seven advisors, more unhappy/at-risk personas, "
            "and more complaints and refunds."
        ),
        manager_question="How does a difficult contact mix affect waits, SLA, and abandonment?",
        expected_behavior="Designed to test pressure from both volume and longer-handling intents.",
        objectives=OBJECTIVES,
        suggested_action="Run the Twin and inspect intent, persona, and queue patterns.",
        risk_level=RiskLevel.HIGH,
        scientific_note="The mixes affect sampled patience and handling; the incident label is context.",
    ),
    "knowledge_failure": ScenarioGuide(
        scenario_key="knowledge_failure",
        title="Knowledge Failure",
        short_description="A normal queue configuration labelled with an outdated knowledge base.",
        situation="Demand, staffing, personas, and requests match Normal Day; KB state is outdated.",
        manager_question="What can the current Twin measure when knowledge quality is degraded?",
        expected_behavior="Queue KPIs should behave like the baseline for the same seed and settings.",
        objectives=OBJECTIVES,
        suggested_action="Use the result to discuss scope; inspect interactions separately for KB quality.",
        risk_level=RiskLevel.MODERATE,
        scientific_note="KB state is context only and does not alter current queue mechanics.",
    ),
    "perfect_storm": ScenarioGuide(
        scenario_key="perfect_storm",
        title="Perfect Storm",
        short_description="Extreme configured demand, difficult contact mix, and low staffing.",
        situation=(
            "Demand is set to 2.50x with four advisors, more unhappy/at-risk personas, "
            "and a complaint/refund-heavy request mix."
        ),
        manager_question="Where does the simulated center break down under compounded pressure?",
        expected_behavior="Designed to test severe overload and identify failed operating targets.",
        objectives=OBJECTIVES,
        suggested_action="Run the Twin, then test additional staffing under the same seed.",
        risk_level=RiskLevel.SEVERE,
        scientific_note=(
            "Delivery disruption and partial-KB labels are context only; configured queue inputs "
            "drive the simulated result."
        ),
    ),
}

SCENARIO_GUIDES: Mapping[str, ScenarioGuide] = MappingProxyType(_GUIDES)


def get_scenario_guide(scenario_key: str) -> ScenarioGuide:
    """Return presentation metadata without changing the scenario configuration."""

    try:
        return SCENARIO_GUIDES[scenario_key]
    except KeyError as exc:
        raise KeyError(f"no scenario guidance for {scenario_key!r}") from exc


def recommended_demo_widget_state() -> dict[str, object]:
    """Return widget values for the demo loader; this never runs the simulation."""

    return {
        "manager_preset": RECOMMENDED_DEMO.scenario_key,
        f"manager_seed_{RECOMMENDED_DEMO.scenario_key}": RECOMMENDED_DEMO.seed,
        f"manager_agents_{RECOMMENDED_DEMO.scenario_key}": RECOMMENDED_DEMO.available_agents,
        "manager_policy_mode": RECOMMENDED_DEMO.policy_mode,
    }


def _target_checks(
    result: SimulationResult, targets: ManagerTargets
) -> tuple[TargetCheck, ...]:
    sla = result.kpis.sla
    abandonment = result.kpis.abandonment_rate
    occupancy = result.kpis.occupancy
    return (
        TargetCheck(
            metric="SLA",
            objective=f">= {targets.minimum_sla:.0%}",
            actual=sla,
            outcome="unavailable"
            if sla is None
            else ("pass" if sla >= targets.minimum_sla else "fail"),
        ),
        TargetCheck(
            metric="Abandonment",
            objective=f"< {targets.maximum_abandonment:.0%}",
            actual=abandonment,
            outcome=(
                "unavailable"
                if abandonment is None
                else ("pass" if abandonment < targets.maximum_abandonment else "fail")
            ),
        ),
        TargetCheck(
            metric="Occupancy",
            objective=f"< {targets.maximum_occupancy:.0%}",
            actual=occupancy,
            outcome=(
                "unavailable"
                if occupancy is None
                else ("pass" if occupancy < targets.maximum_occupancy else "warning")
            ),
        ),
    )


def interpret_simulation_result(
    result: SimulationResult,
    targets: ManagerTargets = DEFAULT_TARGETS,
) -> ResultInterpretation:
    """Explain a result with transparent, deterministic V1 managerial rules."""

    checks = _target_checks(result, targets)
    failed_count = sum(check.outcome in {"fail", "warning"} for check in checks)
    sla = result.kpis.sla
    abandonment = result.kpis.abandonment_rate
    occupancy = result.kpis.occupancy

    severe = (
        (sla is not None and sla < 0.40)
        or (abandonment is not None and abandonment >= 0.40)
        or (occupancy is not None and occupancy >= 0.97)
    )
    if severe:
        status = CenterStatus.CRITICAL
    elif failed_count >= 2:
        status = CenterStatus.OVERLOADED
    elif failed_count == 1:
        status = CenterStatus.UNDER_PRESSURE
    else:
        status = CenterStatus.HEALTHY

    reasons: list[str] = []
    for check in checks:
        if check.outcome == "fail":
            reasons.append(f"{check.metric} misses the V1 target ({check.objective}).")
        elif check.outcome == "warning":
            reasons.append(
                f"{check.metric} is at or above the V1 caution level ({check.objective})."
            )
    average_wait = result.kpis.average_waiting_time
    if average_wait is not None and average_wait > result.sla_target_minutes:
        reasons.append("Average wait is above the simulation's SLA wait threshold.")
    final_backlog = result.snapshots[-1].queue_size if result.snapshots else 0
    if final_backlog > 0:
        reasons.append(
            f"{final_backlog} contacts remain queued at the simulation horizon."
        )
    if not reasons:
        reasons.append("All three V1 managerial targets are met in this simulated run.")

    if status is CenterStatus.HEALTHY:
        next_step = "Open Forecast to check whether capacity remains sufficient over the next 24 hours."
    elif status is CenterStatus.UNDER_PRESSURE:
        next_step = (
            "Open Twin Monitor to inspect when pressure develops, then review capacity."
        )
    else:
        next_step = (
            "Open Compare Decisions and test additional staffing under the same seed."
        )

    return ResultInterpretation(status, tuple(reasons), checks, next_step)
