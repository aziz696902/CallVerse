"""Pure manager-dashboard logic over existing CallVerse domain contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import load_support_profile
from callverse.domain import DomainModel, ScenarioConfig
from callverse.quality import aggregate_quality
from callverse.scenarios import get_scenario
from callverse.simulation import (
    DEFAULT_POLICY,
    SimulationPolicy,
    SimulationResult,
    run_simulation,
)

if TYPE_CHECKING:
    from callverse.customer_advisor import AdvisorInteraction
    from callverse.quality import QualityEvaluationResult

PolicyMode = Literal["prototype", "calibrated"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_PROFILE_PATH = PROJECT_ROOT / "data/processed/calibration/support_center_profile.json"

WARNING_THRESHOLDS = {
    "occupancy": 0.85,
    "abandonment_rate": 0.15,
    "sla": 0.80,
}


class ManagerRun(DomainModel):
    scenario: ScenarioConfig
    policy_mode: PolicyMode
    result: SimulationResult


class DecisionComparison(DomainModel):
    before: ManagerRun
    after: ManagerRun
    changed_field: Literal["available_agents"] = "available_agents"


class KpiCard(DomainModel):
    label: str
    value: str
    raw_value: float | int | None = None


class ComparisonRow(DomainModel):
    metric: str
    before: float | int | None
    after: float | int | None
    delta: float | int | None
    unit: Literal["count", "minutes", "percentage_points"]


def policy_for(mode: PolicyMode) -> SimulationPolicy:
    if mode == "prototype":
        return DEFAULT_POLICY
    if mode == "calibrated":
        return build_calibrated_policy(load_support_profile(SUPPORT_PROFILE_PATH))
    raise ValueError(f"unknown policy mode: {mode}")


def configure_scenario(
    preset_name: str,
    *,
    seed: int,
    available_agents: int,
    demand_multiplier: float,
    duration: float,
) -> ScenarioConfig:
    preset = get_scenario(preset_name)
    return preset.model_copy(
        update={
            "random_seed": seed,
            "available_agents": available_agents,
            "demand_multiplier": demand_multiplier,
            "simulation_duration": duration,
        }
    )


def run_manager_simulation(scenario: ScenarioConfig, policy_mode: PolicyMode) -> ManagerRun:
    result = run_simulation(scenario, seed=scenario.random_seed, policy=policy_for(policy_mode))
    return ManagerRun(scenario=scenario, policy_mode=policy_mode, result=result)


def run_staffing_what_if(before: ManagerRun, after_agents: int) -> DecisionComparison:
    if after_agents <= 0:
        raise ValueError("after-agents value must be positive")
    after_scenario = before.scenario.model_copy(update={"available_agents": after_agents})
    after = run_manager_simulation(after_scenario, before.policy_mode)
    return DecisionComparison(before=before, after=after)


def format_count(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


def format_minutes(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} min"


def format_percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def kpi_cards(result: SimulationResult) -> tuple[KpiCard, ...]:
    return (
        KpiCard(label="Generated contacts", value=format_count(result.counts.generated), raw_value=result.counts.generated),
        KpiCard(label="Completed contacts", value=format_count(result.counts.completed), raw_value=result.counts.completed),
        KpiCard(label="Average wait", value=format_minutes(result.kpis.average_waiting_time), raw_value=result.kpis.average_waiting_time),
        KpiCard(label="SLA", value=format_percentage(result.kpis.sla), raw_value=result.kpis.sla),
        KpiCard(label="Abandonment", value=format_percentage(result.kpis.abandonment_rate), raw_value=result.kpis.abandonment_rate),
        KpiCard(label="Occupancy", value=format_percentage(result.kpis.occupancy), raw_value=result.kpis.occupancy),
        KpiCard(label="Average handling", value=format_minutes(result.kpis.average_handling_time), raw_value=result.kpis.average_handling_time),
    )


def timeline_rows(result: SimulationResult) -> list[dict[str, float | int]]:
    return [
        {
            "simulation_time": snapshot.simulation_time,
            "queue_size": snapshot.queue_size,
            "busy_agents": snapshot.busy_agents,
            "completed": snapshot.completed_count,
            "abandoned": snapshot.abandoned_count,
        }
        for snapshot in result.snapshots
    ]


def intent_mix_rows(result: SimulationResult) -> list[dict[str, str | int]]:
    return [
        {"intent": intent.value, "contacts": count}
        for intent, count in result.intent_counts.items()
    ]


def persona_mix_rows(result: SimulationResult) -> list[dict[str, str | int]]:
    return [
        {"persona": persona.value, "contacts": count}
        for persona, count in result.persona_counts.items()
    ]


def scenario_warnings(result: SimulationResult) -> tuple[str, ...]:
    warnings: list[str] = []
    if result.kpis.occupancy is not None and result.kpis.occupancy > WARNING_THRESHOLDS["occupancy"]:
        warnings.append("Center operating near saturation: occupancy exceeds 85%.")
    if result.kpis.abandonment_rate is not None and result.kpis.abandonment_rate > WARNING_THRESHOLDS["abandonment_rate"]:
        warnings.append("Severe abandonment: more than 15% of contacts abandoned.")
    if result.kpis.sla is not None and result.kpis.sla < WARNING_THRESHOLDS["sla"]:
        warnings.append("Poor SLA attainment: fewer than 80% of served contacts met the target.")
    if result.snapshots and result.snapshots[-1].queue_size > result.snapshots[0].queue_size:
        warnings.append("Queue remains above its starting level at the simulation horizon.")
    return tuple(warnings)


def comparison_rows(comparison: DecisionComparison) -> tuple[ComparisonRow, ...]:
    before = comparison.before.result
    after = comparison.after.result

    def delta(first: float | None, second: float | None, scale: float = 1.0):
        return None if first is None or second is None else round((second - first) * scale, 3)

    return (
        ComparisonRow(
            metric="Generated contacts",
            before=before.counts.generated,
            after=after.counts.generated,
            delta=after.counts.generated - before.counts.generated,
            unit="count",
        ),
        ComparisonRow(
            metric="Average wait",
            before=before.kpis.average_waiting_time,
            after=after.kpis.average_waiting_time,
            delta=delta(before.kpis.average_waiting_time, after.kpis.average_waiting_time),
            unit="minutes",
        ),
        ComparisonRow(
            metric="SLA",
            before=before.kpis.sla,
            after=after.kpis.sla,
            delta=delta(before.kpis.sla, after.kpis.sla, 100),
            unit="percentage_points",
        ),
        ComparisonRow(
            metric="Abandonment",
            before=before.kpis.abandonment_rate,
            after=after.kpis.abandonment_rate,
            delta=delta(before.kpis.abandonment_rate, after.kpis.abandonment_rate, 100),
            unit="percentage_points",
        ),
        ComparisonRow(
            metric="Occupancy",
            before=before.kpis.occupancy,
            after=after.kpis.occupancy,
            delta=delta(before.kpis.occupancy, after.kpis.occupancy, 100),
            unit="percentage_points",
        ),
        ComparisonRow(
            metric="Average handling",
            before=before.kpis.average_handling_time,
            after=after.kpis.average_handling_time,
            delta=delta(before.kpis.average_handling_time, after.kpis.average_handling_time),
            unit="minutes",
        ),
    )


def comparison_table(comparison: DecisionComparison) -> list[dict[str, object]]:
    rows = []
    for row in comparison_rows(comparison):
        if row.unit == "percentage_points":
            before = format_percentage(float(row.before) if row.before is not None else None)
            after = format_percentage(float(row.after) if row.after is not None else None)
            delta_text = "N/A" if row.delta is None else f"{float(row.delta):+.1f} pp"
        elif row.unit == "minutes":
            before = format_minutes(float(row.before) if row.before is not None else None)
            after = format_minutes(float(row.after) if row.after is not None else None)
            delta_text = "N/A" if row.delta is None else f"{float(row.delta):+.2f} min"
        else:
            before = format_count(int(row.before) if row.before is not None else None)
            after = format_count(int(row.after) if row.after is not None else None)
            delta_text = "N/A" if row.delta is None else f"{int(row.delta):+d}"
        rows.append({"KPI": row.metric, "Before": before, "After": after, "Delta": delta_text})
    return rows


def export_payload(
    run: ManagerRun,
    comparison: DecisionComparison | None = None,
) -> dict[str, object]:
    result = run.result
    payload: dict[str, object] = {
        "configuration": {
            "scenario": run.scenario.name,
            "seed": result.seed,
            "policy_mode": run.policy_mode,
            "available_agents": run.scenario.available_agents,
            "demand_multiplier": run.scenario.demand_multiplier,
            "simulation_duration_minutes": run.scenario.simulation_duration,
        },
        "counts": result.counts.model_dump(mode="json"),
        "kpis": result.kpis.model_dump(mode="json"),
        "warnings": list(scenario_warnings(result)),
    }
    if comparison is not None:
        payload["comparison"] = {
            "after_available_agents": comparison.after.scenario.available_agents,
            "same_seed": comparison.before.result.seed == comparison.after.result.seed,
            "simulated_effect": [row.model_dump(mode="json") for row in comparison_rows(comparison)],
        }
    return payload


def export_json(run: ManagerRun, comparison: DecisionComparison | None = None) -> str:
    return json.dumps(export_payload(run, comparison), indent=2)


def interaction_summary(interaction: AdvisorInteraction) -> dict[str, object]:
    metadata = interaction.metadata
    return {
        "predicted_intent": (
            metadata.predicted_intent.value if metadata.predicted_intent is not None else None
        ),
        "classifier_confidence": metadata.classifier_confidence,
        "classifier_accepted": metadata.classifier_accepted,
        "routing_path": metadata.routing_path.value,
        "fallback_reason": metadata.fallback_reason,
        "order_id": metadata.extracted_order_id,
        "tools_used": list(metadata.tools_used),
        "rag_sources": list(metadata.rag_sources),
        "resolved": interaction.result.resolved,
        "escalated": interaction.result.escalated,
    }


def session_quality_summary(
    results: list[QualityEvaluationResult],
) -> dict[str, object] | None:
    if not results:
        return None
    return aggregate_quality(results)
