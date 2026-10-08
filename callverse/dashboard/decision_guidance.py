"""Deterministic presentation logic for same-seed staffing comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .scenario_guidance import (
    DEFAULT_TARGETS,
    CenterStatus,
    ManagerTargets,
    get_scenario_guide,
    interpret_simulation_result,
)
from .view_models import DecisionComparison


class ChangeAssessment(str, Enum):
    BETTER = "better"
    WORSE = "worse"
    NEUTRAL = "neutral"


class DecisionOutcome(str, Enum):
    STRONGLY_IMPROVED = "STRONGLY IMPROVED"
    IMPROVED = "IMPROVED"
    MIXED_TRADE_OFF = "MIXED TRADE-OFF"
    LITTLE_CHANGE = "LITTLE CHANGE"
    WORSENED = "WORSENED"


@dataclass(frozen=True)
class MetricChange:
    key: str
    label: str
    before: float | int
    after: float | int
    delta: float | int
    unit: Literal["percentage_points", "minutes", "count"]
    preference: Literal["higher", "lower", "target"]
    assessment: ChangeAssessment


@dataclass(frozen=True)
class TargetComparison:
    metric: str
    objective: str
    before: str
    after: str


@dataclass(frozen=True)
class DecisionNarrative:
    problem: str
    action: str
    outcome: DecisionOutcome
    conclusion: str
    trade_off: str
    next_step: str
    changes: tuple[MetricChange, ...]
    targets: tuple[TargetComparison, ...]
    changed_parameter: str
    held_constant: tuple[str, ...]


SAME_SEED_EXPLANATION = (
    "Both runs use the same random seed and scenario configuration. This reduces random "
    "variation so the staffing change is easier to compare."
)
PRODUCTION_AB_DISCLAIMER = (
    "This is a simulated controlled comparison, not a production A/B test."
)


def recommended_decision_widget_state(comparison_key: str) -> dict[str, int]:
    """Prefill the official 3-to-5 test without creating or running a comparison."""

    return {comparison_key: 5}


def _rate_assessment(
    before: float, after: float, *, higher_is_better: bool
) -> ChangeAssessment:
    if abs(after - before) < 1e-12:
        return ChangeAssessment.NEUTRAL
    improved = after > before if higher_is_better else after < before
    return ChangeAssessment.BETTER if improved else ChangeAssessment.WORSE


def _occupancy_assessment(before: float, after: float, cap: float) -> ChangeAssessment:
    before_passes = before < cap
    after_passes = after < cap
    if before_passes and not after_passes:
        return ChangeAssessment.WORSE
    if not before_passes and after_passes:
        return ChangeAssessment.BETTER
    if before_passes and after_passes:
        return ChangeAssessment.NEUTRAL
    return _rate_assessment(before, after, higher_is_better=False)


def metric_changes(
    comparison: DecisionComparison,
    targets: ManagerTargets = DEFAULT_TARGETS,
) -> tuple[MetricChange, ...]:
    """Build correctly scaled deltas and explicit metric-direction metadata."""

    before = comparison.before.result
    after = comparison.after.result
    values = (
        (
            "sla",
            "SLA",
            before.kpis.sla,
            after.kpis.sla,
            "percentage_points",
            "higher",
        ),
        (
            "abandonment",
            "Abandonment",
            before.kpis.abandonment_rate,
            after.kpis.abandonment_rate,
            "percentage_points",
            "lower",
        ),
        (
            "average_wait",
            "Average wait",
            before.kpis.average_waiting_time,
            after.kpis.average_waiting_time,
            "minutes",
            "lower",
        ),
        (
            "occupancy",
            "Occupancy",
            before.kpis.occupancy,
            after.kpis.occupancy,
            "percentage_points",
            "target",
        ),
        (
            "completed",
            "Completed contacts",
            before.counts.completed,
            after.counts.completed,
            "count",
            "higher",
        ),
        (
            "final_backlog",
            "Final queue backlog",
            before.snapshots[-1].queue_size if before.snapshots else 0,
            after.snapshots[-1].queue_size if after.snapshots else 0,
            "count",
            "lower",
        ),
    )
    changes: list[MetricChange] = []
    for key, label, before_value, after_value, unit, preference in values:
        if before_value is None or after_value is None:
            continue
        if key == "occupancy":
            assessment = _occupancy_assessment(
                float(before_value), float(after_value), targets.maximum_occupancy
            )
        else:
            assessment = _rate_assessment(
                float(before_value),
                float(after_value),
                higher_is_better=preference == "higher",
            )
        scale = 100 if unit == "percentage_points" else 1
        delta = (after_value - before_value) * scale
        if unit == "count":
            delta = int(delta)
        changes.append(
            MetricChange(
                key=key,
                label=label,
                before=before_value,
                after=after_value,
                delta=delta,
                unit=unit,
                preference=preference,
                assessment=assessment,
            )
        )
    return tuple(changes)


def format_metric_value(change: MetricChange, value: float) -> str:
    if change.unit == "percentage_points":
        return f"{float(value):.1%}"
    if change.unit == "minutes":
        return f"{float(value):.2f} min"
    return f"{int(value):,}"


def format_metric_delta(change: MetricChange) -> str:
    if change.unit == "percentage_points":
        return f"{float(change.delta):+.1f} percentage points"
    if change.unit == "minutes":
        return f"{float(change.delta):+.2f} min"
    return f"{int(change.delta):+d} contacts"


def _target_comparisons(
    comparison: DecisionComparison,
    targets: ManagerTargets,
) -> tuple[TargetComparison, ...]:
    before_checks = interpret_simulation_result(
        comparison.before.result, targets
    ).target_checks
    after_checks = interpret_simulation_result(
        comparison.after.result, targets
    ).target_checks
    return tuple(
        TargetComparison(
            metric=before.metric,
            objective=before.objective,
            before=before.outcome,
            after=after.outcome,
        )
        for before, after in zip(before_checks, after_checks, strict=True)
    )


def _material_assessments(changes: tuple[MetricChange, ...]) -> tuple[int, int]:
    thresholds = {"sla": 2.0, "abandonment": 2.0, "average_wait": 0.25}
    improved = 0
    worsened = 0
    for change in changes:
        threshold = thresholds.get(change.key)
        if threshold is None or abs(float(change.delta)) < threshold:
            continue
        if change.assessment is ChangeAssessment.BETTER:
            improved += 1
        elif change.assessment is ChangeAssessment.WORSE:
            worsened += 1
    return improved, worsened


def _classify_outcome(
    comparison: DecisionComparison,
    changes: tuple[MetricChange, ...],
    targets: ManagerTargets,
) -> DecisionOutcome:
    before_status = interpret_simulation_result(
        comparison.before.result, targets
    ).status
    after_status = interpret_simulation_result(comparison.after.result, targets).status
    rank = {
        CenterStatus.HEALTHY: 0,
        CenterStatus.UNDER_PRESSURE: 1,
        CenterStatus.OVERLOADED: 2,
        CenterStatus.CRITICAL: 3,
    }
    improved, worsened = _material_assessments(changes)
    if worsened >= 2 and improved == 0:
        return DecisionOutcome.WORSENED
    if (
        before_status is not CenterStatus.HEALTHY
        and after_status is CenterStatus.HEALTHY
        and improved >= 2
        and worsened == 0
    ):
        return DecisionOutcome.STRONGLY_IMPROVED
    if (
        rank[after_status] < rank[before_status]
        and after_status in {CenterStatus.HEALTHY, CenterStatus.UNDER_PRESSURE}
        and improved >= 1
        and worsened == 0
    ):
        return DecisionOutcome.IMPROVED
    if improved and (
        worsened or after_status in {CenterStatus.OVERLOADED, CenterStatus.CRITICAL}
    ):
        return DecisionOutcome.MIXED_TRADE_OFF
    if worsened and not improved:
        return DecisionOutcome.WORSENED
    if improved:
        return DecisionOutcome.IMPROVED
    return DecisionOutcome.LITTLE_CHANGE


def _conclusion(outcome: DecisionOutcome) -> str:
    return {
        DecisionOutcome.STRONGLY_IMPROVED: (
            "The tested staffing change stabilizes the simulated center: SLA reaches target "
            "and abandonment falls below the V1 threshold."
        ),
        DecisionOutcome.IMPROVED: (
            "The tested configuration materially improves service under identical seeded "
            "conditions, although the remaining target checks still matter."
        ),
        DecisionOutcome.MIXED_TRADE_OFF: (
            "The change improves part of the service picture, but important operating targets "
            "remain unresolved or another primary metric worsens."
        ),
        DecisionOutcome.LITTLE_CHANGE: (
            "The tested configuration produces little material change in the primary service "
            "metrics."
        ),
        DecisionOutcome.WORSENED: (
            "The tested configuration worsens the primary operational targets under these "
            "simulated conditions."
        ),
    }[outcome]


def build_decision_narrative(
    comparison: DecisionComparison,
    targets: ManagerTargets = DEFAULT_TARGETS,
) -> DecisionNarrative:
    """Build the Problem → Action → Effect → Conclusion story without an LLM."""

    before = comparison.before
    after = comparison.after
    guide = get_scenario_guide(before.scenario.name)
    before_status = interpret_simulation_result(
        before.result, targets
    ).status.value.lower()
    status_article = "an" if before_status[0] in "aeiou" else "a"
    changes = metric_changes(comparison, targets)
    outcome = _classify_outcome(comparison, changes, targets)
    agent_delta = after.scenario.available_agents - before.scenario.available_agents
    if agent_delta > 0:
        action = (
            f"Increase staffing from {before.scenario.available_agents} to "
            f"{after.scenario.available_agents} agents."
        )
        trade_off = (
            f"Service changes are achieved with {agent_delta} additional "
            f"agent{'s' if agent_delta != 1 else ''}; no monetary cost is inferred."
        )
    elif agent_delta < 0:
        action = (
            f"Reduce staffing from {before.scenario.available_agents} to "
            f"{after.scenario.available_agents} agents."
        )
        trade_off = (
            f"The tested configuration uses {abs(agent_delta)} fewer "
            f"agent{'s' if abs(agent_delta) != 1 else ''}; service effects must be reviewed."
        )
    else:
        action = (
            f"Keep staffing unchanged at {before.scenario.available_agents} agents."
        )
        trade_off = (
            "No staffing trade-off was introduced because agent count did not change."
        )

    after_status = interpret_simulation_result(after.result, targets).status
    if outcome is DecisionOutcome.MIXED_TRADE_OFF:
        next_step = "Next: compare the service improvement against the additional staffing required."
    elif after_status is CenterStatus.HEALTHY:
        next_step = (
            "Next: inspect Forecast and Workforce to see whether this staffing level remains "
            "sufficient across the upcoming 24-hour demand pattern."
        )
    elif outcome is DecisionOutcome.LITTLE_CHANGE:
        next_step = (
            "Next: test a meaningfully different staffing level or inspect Workforce."
        )
    elif outcome is DecisionOutcome.WORSENED:
        next_step = "Next: restore capacity and test a higher staffing level under the same seed."
    else:
        next_step = (
            "Next: test another staffing level or inspect Workforce recommendations."
        )

    return DecisionNarrative(
        problem=(
            f"{guide.title} with {before.scenario.available_agents} agents shows "
            f"{status_article} "
            f"{before_status} simulated center."
        ),
        action=action,
        outcome=outcome,
        conclusion=_conclusion(outcome),
        trade_off=trade_off,
        next_step=next_step,
        changes=changes,
        targets=_target_comparisons(comparison, targets),
        changed_parameter=(
            f"Available agents: {before.scenario.available_agents} → "
            f"{after.scenario.available_agents}"
        ),
        held_constant=(
            f"Scenario: {guide.title}",
            f"Seed: {before.result.seed}",
            f"Simulator mode: {before.policy_mode}",
            f"Duration: {before.scenario.simulation_duration:g} simulated minutes",
            f"Demand multiplier: {before.scenario.demand_multiplier:g}x",
            "Persona mix, request mix, and all other scenario settings",
        ),
    )
