"""Independent multi-seed Erlang-C versus Digital Twin validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean

from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import load_support_profile
from callverse.scenarios import get_scenario
from callverse.simulation import run_simulation

from .erlang_c import evaluate_erlang_c, minimum_agents
from .manager import SUPPORT_PROFILE_PATH, default_workforce_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data/processed/workforce/erlang_c_validation.json"
SEEDS = (101, 202, 303, 404, 505)
CASE_CONTACTS = {
    "low_demand": 5.0,
    "normal_demand": 15.0,
    "high_demand": 25.0,
    "near_saturation": 35.0,
}


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def build_validation_report() -> dict[str, object]:
    config = default_workforce_config(forecast_buffer_percent=0, reduction_hold_intervals=1)
    profile = load_support_profile(SUPPORT_PROFILE_PATH)
    policy = build_calibrated_policy(profile)
    cases = []
    for name, contacts_per_half_hour in CASE_CONTACTS.items():
        arrival_rate = contacts_per_half_hour * 2
        search = minimum_agents(
            arrival_rate_per_hour=arrival_rate,
            service_rate_per_hour_per_agent=config.service_rate_per_hour,
            sla_wait_threshold_minutes=config.sla_wait_threshold_minutes,
            target_service_level=config.target_service_level,
            max_occupancy=config.max_occupancy,
            min_agents=config.min_agents,
            max_agents=config.max_agents,
        )
        agents = (
            max(1, math.ceil(search.metrics.offered_load))
            if name == "near_saturation"
            else search.agents
        )
        analytical = evaluate_erlang_c(
            arrival_rate,
            config.service_rate_per_hour,
            agents,
            config.sla_wait_threshold_minutes,
        )
        results = []
        for seed in SEEDS:
            scenario = get_scenario("normal_day").model_copy(
                update={
                    "random_seed": seed,
                    "available_agents": agents,
                    "demand_multiplier": arrival_rate / 60 / policy.base_arrival_rate_per_minute,
                    "simulation_duration": 480.0,
                }
            )
            results.append(run_simulation(scenario, seed=seed, policy=policy))
        cases.append(
            {
                "case": name,
                "forecast_contacts_per_half_hour": contacts_per_half_hour,
                "arrival_rate_per_hour": arrival_rate,
                "agents": agents,
                "staffing_basis": (
                    "minimum Erlang-C recommendation"
                    if name != "near_saturation"
                    else "smallest stable integer above offered load; intentionally near saturation"
                ),
                "erlang_c": analytical.model_dump(mode="json"),
                "digital_twin": {
                    "replications": len(results),
                    "seeds": list(SEEDS),
                    "mean_generated": fmean(result.counts.generated for result in results),
                    "mean_average_wait_minutes": _mean_optional(
                        [result.kpis.average_waiting_time for result in results]
                    ),
                    "mean_sla": _mean_optional([result.kpis.sla for result in results]),
                    "mean_occupancy": _mean_optional(
                        [result.kpis.occupancy for result in results]
                    ),
                    "mean_abandonment": _mean_optional(
                        [result.kpis.abandonment_rate for result in results]
                    ),
                    "mean_aht_minutes": _mean_optional(
                        [result.kpis.average_handling_time for result in results]
                    ),
                },
            }
        )
    return {
        "version": "callverse-workforce-validation-v1",
        "method": "fixed-capacity representative intervals; calibrated Digital Twin",
        "dynamic_staffing": "deferred",
        "configuration": {
            **config.model_dump(mode="json"),
            "service_rate_per_hour_per_agent": config.service_rate_per_hour,
            "simulation_duration_minutes": 480,
            "seeds": list(SEEDS),
        },
        "scientific_note": (
            "Erlang-C is an M/M/c no-abandonment approximation. The authoritative CallVerse "
            "Twin uses empirical service/patience distributions, abandonment, and a varying "
            "arrival profile, so disagreement is expected and calibration is not altered."
        ),
        "cases": cases,
    }


def main() -> None:
    report = build_validation_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
