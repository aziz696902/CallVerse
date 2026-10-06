"""Cross-check the compact RL environment against frozen Digital Twin evidence."""

from __future__ import annotations

import json
from statistics import fmean

import numpy as np
import pandas as pd

from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import load_support_profile
from callverse.scenarios import get_scenario
from callverse.simulation import run_simulation
from callverse.workforce.manager import SUPPORT_PROFILE_PATH

from .env import PROJECT_ROOT, WorkforceEnv
from .evaluation import fixed_policy, run_episode

WORKFORCE_VALIDATION_PATH = (
    PROJECT_ROOT / "data/processed/workforce/erlang_c_validation.json"
)
OUTPUT_PATH = PROJECT_ROOT / "data/processed/rl/environment_crosscheck.json"


def build_crosscheck() -> dict[str, object]:
    workforce = json.loads(WORKFORCE_VALIDATION_PATH.read_text(encoding="utf-8"))
    twin_policy = build_calibrated_policy(load_support_profile(SUPPORT_PROFILE_PATH))
    cases = []
    for case in workforce["cases"]:
        contacts = float(case["forecast_contacts_per_half_hour"])
        episode = [
            (
                pd.Timestamp("2000-01-01"),
                np.full(48, contacts, dtype=np.float32),
                np.full(48, contacts, dtype=np.float32),
            )
        ]
        env = WorkforceEnv(split="test", episodes=episode)
        outcomes = [
            run_episode(env, fixed_policy(int(case["agents"])), 0, seed)[0]
            for seed in workforce["configuration"]["seeds"]
        ]
        twin_outcomes = []
        for seed in workforce["configuration"]["seeds"]:
            scenario = get_scenario("normal_day").model_copy(
                update={
                    "random_seed": seed,
                    "available_agents": int(case["agents"]),
                    "demand_multiplier": (
                        float(case["arrival_rate_per_hour"])
                        / 60
                        / twin_policy.base_arrival_rate_per_minute
                    ),
                    "simulation_duration": 480.0,
                }
            )
            twin_outcomes.append(run_simulation(scenario, seed=seed, policy=twin_policy))
        cases.append(
            {
                "case": case["case"],
                "contacts_per_half_hour": contacts,
                "agents": case["agents"],
                "seeds": workforce["configuration"]["seeds"],
                "rl_environment": {
                    "mean_completed": fmean(row["completed"] for row in outcomes),
                    "mean_completed_per_hour": fmean(
                        row["completed"] / 24 for row in outcomes
                    ),
                    "mean_wait_minutes": fmean(row["average_wait_minutes"] for row in outcomes),
                    "mean_service_level": fmean(row["service_level"] for row in outcomes),
                    "mean_occupancy": fmean(row["average_occupancy"] for row in outcomes),
                    "mean_abandonment": fmean(row["abandonment_rate"] for row in outcomes),
                },
                "digital_twin": {
                    **case["digital_twin"],
                    "mean_completed": fmean(
                        result.counts.completed for result in twin_outcomes
                    ),
                    "mean_completed_per_hour": fmean(
                        result.counts.completed / 8 for result in twin_outcomes
                    ),
                },
            }
        )
    return {
        "version": "callverse-rl-environment-crosscheck-v1",
        "purpose": "directional validation only; the RL environment is not the Digital Twin",
        "horizon_note": "RL cases use 24 hours; frozen Twin cases use 8 hours",
        "cases": cases,
    }


def main() -> None:
    report = build_crosscheck()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
