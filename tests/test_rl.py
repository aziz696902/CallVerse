"""Phase 11 checks; none of these tests performs full PPO training."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from callverse.dashboard.manager import rl_artifacts_available
from callverse.rl.env import PROJECT_ROOT, WorkforceEnv, load_demand_episodes
from callverse.rl.evaluation import (
    erlang_policy,
    fixed_policy,
    ppo_policy,
    run_episode,
    summarize,
)
from callverse.rl.models import RewardWeights


def constant_episode(contacts: float = 20.0):
    return [
        (
            pd.Timestamp("2000-01-01"),
            np.full(48, contacts, dtype=np.float32),
            np.full(48, contacts, dtype=np.float32),
        )
    ]


def test_environment_passes_stable_baselines_checker():
    check_env(WorkforceEnv(episodes=constant_episode()), warn=True)


def test_episode_has_exactly_48_half_hour_decisions():
    env = WorkforceEnv(episodes=constant_episode())
    observation, _ = env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        observation, _, done, truncated, _ = env.step(2)
        assert not truncated
        assert observation.shape == (10,)
        steps += 1
    assert steps == 48


def test_action_zero_maps_to_one_agent_and_bounds_are_enforced():
    env = WorkforceEnv(episodes=constant_episode())
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert info["agents"] == 1
    with pytest.raises(ValueError, match="invalid staffing action"):
        env.step(env.action_space.n)


def test_reset_and_transition_are_deterministic_for_same_seed():
    left = WorkforceEnv(episodes=constant_episode())
    right = WorkforceEnv(episodes=constant_episode())
    left_observation, _ = left.reset(seed=123)
    right_observation, _ = right.reset(seed=123)
    assert np.array_equal(left_observation, right_observation)
    assert left.step(4)[1:] == right.step(4)[1:]


def test_historical_forecast_signal_uses_only_seven_day_lag():
    episodes = load_demand_episodes("test")
    demand = pd.read_csv(
        PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv",
        parse_dates=["timestamp"],
    ).set_index("timestamp")["contacts"]
    date, _, forecast = episodes[0]
    expected = demand.loc[
        date - pd.Timedelta(days=7) : date - pd.Timedelta(days=7) + pd.Timedelta(hours=23.5)
    ].to_numpy()
    assert np.array_equal(forecast, expected)


def test_observation_does_not_reveal_realized_future_demand():
    forecast = np.full(48, 7, dtype=np.float32)
    first = [(pd.Timestamp("2000-01-01"), np.full(48, 1, dtype=np.float32), forecast)]
    second = [(pd.Timestamp("2000-01-01"), np.full(48, 99, dtype=np.float32), forecast)]
    first_observation, _ = WorkforceEnv(episodes=first).reset(seed=8)
    second_observation, _ = WorkforceEnv(episodes=second).reset(seed=8)
    assert np.array_equal(first_observation, second_observation)


def test_abandonment_penalty_prevents_rewarding_understaffing_exploit():
    low = WorkforceEnv(episodes=constant_episode(80))
    high = WorkforceEnv(episodes=constant_episode(80))
    low.reset(seed=5)
    high.reset(seed=5)
    low_reward = low.step(0)[1]
    high_reward = high.step(19)[1]
    assert high_reward > low_reward


def test_staffing_cost_penalty_prefers_fewer_agents_when_other_terms_are_zero():
    weights = RewardWeights(
        service_level=0, waiting=0, abandonment=0, staffing=1, overload=0, staffing_change=0
    )
    low = WorkforceEnv(episodes=constant_episode(0), reward_weights=weights)
    high = WorkforceEnv(episodes=constant_episode(0), reward_weights=weights)
    low.reset(seed=5)
    high.reset(seed=5)
    assert low.step(0)[1] > high.step(19)[1]


def test_staffing_change_penalty_is_applied_independently():
    weights = RewardWeights(
        service_level=0, waiting=0, abandonment=0, staffing=0, overload=0, staffing_change=1
    )
    steady = WorkforceEnv(episodes=constant_episode(0), reward_weights=weights)
    changed = WorkforceEnv(episodes=constant_episode(0), reward_weights=weights)
    steady.reset(seed=5)
    changed.reset(seed=5)
    assert steady.step(0)[1] > changed.step(1)[1]


def test_backlog_carries_between_intervals():
    class SlowService:
        def sample(self, _rng):
            return 10_000.0

    env = WorkforceEnv(episodes=constant_episode(1))
    env.service_sampler = SlowService()
    env.patience_sampler = type("Patient", (), {"sample": lambda self, rng: 100.0})()
    env.reset(seed=2)
    _, _, _, _, first = env.step(0)
    _, _, _, _, second = env.step(0)
    assert first["backlog"] == 1
    assert second["backlog"] == 2


def test_selected_ppo_artifact_loads_and_runs_full_episode():
    model = PPO.load(PROJECT_ROOT / "models/ppo_workforce/ppo_policy.zip", device="cpu")
    metrics, trajectory = run_episode(
        WorkforceEnv(split="test"), ppo_policy(model), episode_index=0, seed=101, keep_trajectory=True
    )
    assert len(trajectory) == 48
    assert metrics["total_contacts"] == metrics["completed"] + metrics["abandoned"] + metrics["remaining"]


def test_selected_ppo_inference_is_deterministic():
    model = PPO.load(PROJECT_ROOT / "models/ppo_workforce/ppo_policy.zip", device="cpu")
    env = WorkforceEnv(split="test")
    observation, _ = env.reset(seed=101, options={"episode_index": 0})
    first = ppo_policy(model)(observation, env)
    second = ppo_policy(model)(observation, env)
    assert first == second


def test_training_metadata_records_three_real_seeds_and_median_selection():
    metadata = json.loads(
        (PROJECT_ROOT / "models/ppo_workforce/metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["training_seeds"] == [17, 29, 43]
    assert metadata["timesteps_per_seed"] == 50_000
    assert metadata["selection_rule"] == "median validation mean reward"
    assert metadata["total_runtime_seconds"] > 0


def test_heldout_comparison_uses_identical_cases_and_seeds():
    report = json.loads(
        (PROJECT_ROOT / "data/processed/rl/policy_evaluation.json").read_text(encoding="utf-8")
    )
    assert report["held_out_split"] == "Phase 9 chronological test split"
    assert report["stochastic_seeds"] == [101, 202, 303]
    assert set(report["strategies"]) == {"fixed", "erlang_c", "ppo"}
    assert all(value["episodes"] == 36 for value in report["strategies"].values())


def test_fixed_policy_is_an_explicit_constant_action():
    env = WorkforceEnv(episodes=constant_episode())
    observation, _ = env.reset(seed=1)
    assert fixed_policy(5)(observation, env) == 4


def test_erlang_adapter_returns_a_bounded_action():
    env = WorkforceEnv(episodes=constant_episode())
    observation, _ = env.reset(seed=1)
    action = erlang_policy()(observation, env)
    assert env.action_space.contains(action)


def test_business_metric_aggregation_reports_mean_and_standard_deviation():
    rows = [
        {
            "reward": 1,
            "abandonment_rate": 0.1,
            "service_level": 0.8,
            "average_wait_minutes": 2,
            "average_occupancy": 0.5,
            "total_agent_hours": 10,
            "average_staffing": 2,
            "peak_staffing": 3,
            "staffing_changes": 1,
        },
        {
            "reward": 3,
            "abandonment_rate": 0.3,
            "service_level": 0.6,
            "average_wait_minutes": 4,
            "average_occupancy": 0.7,
            "total_agent_hours": 20,
            "average_staffing": 4,
            "peak_staffing": 5,
            "staffing_changes": 3,
        },
    ]
    report = summarize(rows)
    assert report["metrics"]["reward"] == {"mean": 2.0, "standard_deviation": 1.0}


def test_dashboard_artifact_gate_handles_present_and_absent_paths(tmp_path):
    existing = tmp_path / "model.zip"
    existing.write_bytes(b"model")
    assert rl_artifacts_available((existing,))
    assert not rl_artifacts_available((tmp_path / "missing.zip",))


def test_crosscheck_keeps_rl_environment_and_twin_results_separate():
    report = json.loads(
        (PROJECT_ROOT / "data/processed/rl/environment_crosscheck.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["cases"]) == 4
    assert all("rl_environment" in case and "digital_twin" in case for case in report["cases"])
