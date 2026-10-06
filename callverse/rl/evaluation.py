"""Fair, common-seed evaluation for workforce staffing strategies."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
from stable_baselines3 import PPO

from callverse.workforce.erlang_c import minimum_agents
from callverse.workforce.manager import default_workforce_config

from .env import PROJECT_ROOT, WorkforceEnv

Policy = Callable[[np.ndarray, WorkforceEnv], int]
OUTPUT_PATH = PROJECT_ROOT / "data/processed/rl/policy_evaluation.json"
MODEL_PATH = PROJECT_ROOT / "models/ppo_workforce/ppo_policy.zip"
EVALUATION_SEEDS = (101, 202, 303)


def fixed_policy(agents: int) -> Policy:
    def choose(_observation: np.ndarray, env: WorkforceEnv) -> int:
        return agents - env.config.min_agents

    return choose


def erlang_policy() -> Policy:
    config = default_workforce_config(forecast_buffer_percent=10, reduction_hold_intervals=1)

    def choose(_observation: np.ndarray, env: WorkforceEnv) -> int:
        slot = env._step  # evaluation adapter reads only the currently available forecast slot
        forecast_contacts = float(env.forecast_signal[min(slot, 47)]) * 1.10
        result = minimum_agents(
            arrival_rate_per_hour=forecast_contacts * 2,
            service_rate_per_hour_per_agent=config.service_rate_per_hour,
            sla_wait_threshold_minutes=config.sla_wait_threshold_minutes,
            target_service_level=config.target_service_level,
            max_occupancy=config.max_occupancy,
            min_agents=env.config.min_agents,
            max_agents=env.config.max_agents,
        )
        return result.agents - env.config.min_agents

    return choose


def ppo_policy(model: PPO) -> Policy:
    def choose(observation: np.ndarray, _env: WorkforceEnv) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return choose


def run_episode(
    env: WorkforceEnv,
    policy: Policy,
    episode_index: int,
    seed: int,
    keep_trajectory: bool = False,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    observation, _ = env.reset(seed=seed, options={"episode_index": episode_index})
    done = False
    trajectory = []
    final_info: dict = {}
    while not done:
        action = policy(observation, env)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info
        if keep_trajectory:
            trajectory.append(
                {
                    "slot": info["slot"],
                    "agents": info["agents"],
                    "realized_contacts": info["arrivals"],
                    "forecast_contacts": round(info["forecast_contacts"], 4),
                    "backlog": info["backlog"],
                    "occupancy": round(info["occupancy"], 6),
                    "service_level": round(info["service_level"], 6),
                    "abandoned": info["abandoned"],
                }
            )
    totals = final_info["episode_totals"]
    completed = int(totals["completed"])
    contacts = int(totals["contacts"])
    abandoned = int(totals["abandoned"])
    return (
        {
            "date": final_info["date"],
            "seed": seed,
            "reward": totals["reward"],
            "total_contacts": contacts,
            "completed": completed,
            "abandoned": abandoned,
            "remaining": contacts - completed - abandoned,
            "abandonment_rate": abandoned / max(contacts, 1),
            "service_level": totals["sla_completed"] / max(completed, 1),
            "average_wait_minutes": totals["wait_sum"] / max(completed, 1),
            "average_occupancy": totals["occupancy_sum"] / env.config.horizon,
            "total_agent_hours": totals["agent_hours"],
            "average_staffing": totals["staffing_sum"] / env.config.horizon,
            "peak_staffing": int(totals["peak_staffing"]),
            "staffing_changes": int(totals["staffing_changes"]),
        },
        trajectory,
    )


def summarize(episodes: list[dict]) -> dict[str, object]:
    fields = (
        "reward",
        "abandonment_rate",
        "service_level",
        "average_wait_minutes",
        "average_occupancy",
        "total_agent_hours",
        "average_staffing",
        "peak_staffing",
        "staffing_changes",
    )
    return {
        "episodes": len(episodes),
        "metrics": {
            field: {
                "mean": fmean(float(row[field]) for row in episodes),
                "standard_deviation": pstdev(float(row[field]) for row in episodes),
            }
            for field in fields
        },
    }


def evaluate_policy(policy: Policy, episode_indices: list[int], seeds=EVALUATION_SEEDS):
    env = WorkforceEnv(split="test")
    results = []
    trajectory = []
    for episode_index in episode_indices:
        for seed in seeds:
            result, rows = run_episode(
                env,
                policy,
                episode_index,
                seed,
                keep_trajectory=not trajectory,
            )
            results.append(result)
            if rows and not trajectory:
                trajectory = rows
    return results, trajectory


def build_comparison(model_path: str | Path = MODEL_PATH) -> dict[str, object]:
    test_env = WorkforceEnv(split="test")
    episode_indices = list(range(min(12, len(test_env.episodes))))
    erlang = erlang_policy()
    erlang_actions = []
    for index in episode_indices:
        observation, _ = test_env.reset(seed=0, options={"episode_index": index})
        for _ in range(48):
            action = erlang(observation, test_env)
            erlang_actions.append(action + test_env.config.min_agents)
            observation, _, done, _, _ = test_env.step(action)
            if done:
                break
    fixed_agents = max(1, round(fmean(erlang_actions)))
    model = PPO.load(str(model_path), device="cpu")
    policies = {
        "fixed": fixed_policy(fixed_agents),
        "erlang_c": erlang,
        "ppo": ppo_policy(model),
    }
    report: dict[str, object] = {
        "version": "callverse-rl-evaluation-v1",
        "experimental": True,
        "held_out_split": "Phase 9 chronological test split",
        "forecast_signal": "seven-day seasonal lag; past observations only",
        "episode_indices": episode_indices,
        "stochastic_seeds": list(EVALUATION_SEEDS),
        "fixed_agents": fixed_agents,
        "strategies": {},
    }
    representative = {}
    for name, policy in policies.items():
        results, trajectory = evaluate_policy(policy, episode_indices)
        report["strategies"][name] = {**summarize(results), "raw_episodes": results}
        representative[name] = trajectory
    report["representative_trajectory"] = representative
    return report


def main() -> None:
    report = build_comparison()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: value["metrics"] for name, value in report["strategies"].items()}, indent=2))


if __name__ == "__main__":
    main()
