"""Bounded CPU training entry point for the experimental PPO policy."""

from __future__ import annotations

import argparse
import json
import platform
import time
from statistics import fmean

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .env import PROJECT_ROOT, WorkforceEnv
from .evaluation import ppo_policy, run_episode
from .models import RewardWeights

MODEL_DIRECTORY = PROJECT_ROOT / "models/ppo_workforce"
MODEL_PATH = MODEL_DIRECTORY / "ppo_policy.zip"
METADATA_PATH = MODEL_DIRECTORY / "metadata.json"
TRAINING_SEEDS = (17, 29, 43)


class TrainingProgressCallback(BaseCallback):
    """Emit lightweight heartbeat messages without adding progress-bar dependencies."""

    def __init__(self, seed: int, total_timesteps: int, report_every: int = 5_000) -> None:
        super().__init__(verbose=0)
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.report_every = report_every
        self.next_report = report_every
        self.started = 0.0

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_report:
            completed = min(self.num_timesteps, self.total_timesteps)
            percent = completed / self.total_timesteps * 100
            elapsed = time.perf_counter() - self.started
            print(
                f"[RUNNING] seed={self.seed} progress={completed:,}/{self.total_timesteps:,} "
                f"({percent:.0f}%) elapsed={elapsed:.1f}s",
                flush=True,
            )
            self.next_report += self.report_every
        return True


def make_model(seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        Monitor(WorkforceEnv(split="train")),
        seed=seed,
        device="cpu",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )


def validation_reward(model: PPO) -> tuple[float, list[float]]:
    env = WorkforceEnv(split="validation")
    indices = list(range(min(8, len(env.episodes))))
    rewards = []
    for index in indices:
        for seed in (901, 902):
            metrics, _ = run_episode(env, ppo_policy(model), index, seed)
            rewards.append(float(metrics["reward"]))
    return fmean(rewards), rewards


def train(timesteps: int, seeds: tuple[int, ...] = TRAINING_SEEDS, save: bool = True):
    started = time.perf_counter()
    candidates = []
    print(
        f"[START] PPO workforce training: {len(seeds)} seed(s), "
        f"{timesteps:,} timesteps per seed, CPU only.",
        flush=True,
    )
    for position, seed in enumerate(seeds, start=1):
        print(f"[SEED {position}/{len(seeds)}] Building model for seed {seed}...", flush=True)
        model = make_model(seed)
        seed_started = time.perf_counter()
        model.learn(
            total_timesteps=timesteps,
            progress_bar=False,
            callback=TrainingProgressCallback(seed, timesteps),
        )
        runtime = time.perf_counter() - seed_started
        print(f"[VALIDATING] seed={seed} on the chronological validation split...", flush=True)
        mean_reward, rewards = validation_reward(model)
        candidates.append(
            {
                "seed": seed,
                "runtime_seconds": runtime,
                "validation_mean_reward": mean_reward,
                "validation_rewards": rewards,
                "model": model,
            }
        )
        print(
            f"[SEED COMPLETE] seed={seed} runtime={runtime:.2f}s "
            f"validation_reward={mean_reward:.3f}",
            flush=True,
        )

    ranked = sorted(candidates, key=lambda row: row["validation_mean_reward"])
    selected = ranked[len(ranked) // 2]  # representative median, not the luckiest seed
    print(
        f"[SELECTED] seed={selected['seed']} by median validation mean reward.",
        flush=True,
    )
    metadata = {
        "version": "callverse-ppo-workforce-v1",
        "experimental": True,
        "algorithm": "Stable-Baselines3 PPO",
        "timesteps_per_seed": timesteps,
        "training_seeds": list(seeds),
        "selection_rule": "median validation mean reward",
        "selected_seed": selected["seed"],
        "total_runtime_seconds": time.perf_counter() - started,
        "candidate_results": [
            {key: value for key, value in row.items() if key != "model"} for row in candidates
        ],
        "policy_network": [32, 32],
        "ppo_hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
        },
        "environment_version": "callverse-workforce-rl-env-v1",
        "observation_schema_version": "callverse-workforce-observation-v1",
        "action_schema_version": "absolute-agents-1-to-20-v1",
        "reward_version": "callverse-workforce-reward-v1",
        "device": "cpu",
        "training_split": "Phase 9 chronological training split",
        "validation_split": "Phase 9 chronological validation split",
        "forecast_signal": "seven-day seasonal lag; past observations only",
        "reward_weights": RewardWeights().to_dict(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    if save:
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        selected["model"].save(MODEL_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[SAVED] PPO model: {MODEL_PATH}", flush=True)
        print(f"[SAVED] Training metadata: {METADATA_PATH}", flush=True)
    print(
        f"[SUCCESS] PPO training finished in {metadata['total_runtime_seconds']:.2f}s.",
        flush=True,
    )
    return selected["model"], metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true", help="train one seed without saving")
    args = parser.parse_args()
    seeds = (TRAINING_SEEDS[0],) if args.smoke else TRAINING_SEEDS
    _, metadata = train(args.timesteps, seeds=seeds, save=not args.smoke)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
