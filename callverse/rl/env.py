"""Gymnasium workforce environment backed by CallVerse's historical demand and calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from callverse.calibration.profiles import load_support_profile
from callverse.simulation.policies import QuantileMinutes

from .models import EnvironmentConfig, RewardWeights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMAND_PATH = PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv"
FORECAST_EVALUATION_PATH = PROJECT_ROOT / "data/processed/forecasting/evaluation.json"
SUPPORT_PROFILE_PATH = PROJECT_ROOT / "data/processed/calibration/support_center_profile.json"


def load_demand_episodes(split: str) -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    """Return complete daily episodes and a leakage-safe lag-7-day forecast signal."""

    report = json.loads(FORECAST_EVALUATION_PATH.read_text(encoding="utf-8"))
    if split not in report["splits"]:
        raise ValueError(f"unknown split: {split}")
    start, end = (pd.Timestamp(value) for value in report["splits"][split])
    frame = pd.read_csv(DEMAND_PATH, parse_dates=["timestamp"]).set_index("timestamp")
    contacts = frame["contacts"].astype(float)
    subset = contacts.loc[start:end]
    episodes = []
    for day, values in subset.groupby(subset.index.date):
        if len(values) != 48:
            continue
        timestamps = values.index
        lagged = timestamps - pd.Timedelta(days=7)
        if not all(timestamp in contacts.index for timestamp in lagged):
            continue
        forecast = contacts.loc[lagged].to_numpy(dtype=np.float32)
        episodes.append((pd.Timestamp(day), values.to_numpy(dtype=np.float32), forecast))
    if not episodes:
        raise ValueError(f"no complete 48-slot episodes for {split}")
    return episodes


class WorkforceEnv(gym.Env[np.ndarray, int]):
    """Choose absolute staffing for 48 half-hour slots.

    Observation order: backlog, current staffing, current lag-7-day forecast,
    next forecast, next-two-hour mean forecast, previous occupancy, previous SLA,
    previous abandonment rate, and sine/cosine time-of-day. All non-cyclical
    fields are clipped to [0, 1]. The policy never observes realized future demand.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        config: EnvironmentConfig | None = None,
        reward_weights: RewardWeights | None = None,
        episodes: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.config = config or EnvironmentConfig()
        self.reward_weights = reward_weights or RewardWeights()
        self.episodes = episodes or load_demand_episodes(split)
        self.action_space = spaces.Discrete(
            self.config.max_agents - self.config.min_agents + 1
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)

        profile = load_support_profile(SUPPORT_PROFILE_PATH)
        self.service_sampler = QuantileMinutes(
            profile.service_time_quantiles.probabilities,
            profile.service_time_quantiles.values,
        )
        self.patience_sampler = QuantileMinutes(
            profile.patience_quantiles.probabilities,
            tuple(max(0.01, value) for value in profile.patience_quantiles.values),
        )
        self._mean_aht = float(profile.service_time_minutes.mean or 3.18)
        self._rng = Random(0)
        self._episode_index = 0
        self._step = 0
        self._queue: list[tuple[float, float]] = []
        self._staffing = self.config.min_agents
        self._last_occupancy = 0.0
        self._last_sla = 1.0
        self._last_abandonment = 0.0
        self._totals: dict[str, float] = {}

    @property
    def current_date(self) -> str:
        return str(self.episodes[self._episode_index][0].date())

    @property
    def realized_demand(self) -> np.ndarray:
        return self.episodes[self._episode_index][1]

    @property
    def forecast_signal(self) -> np.ndarray:
        return self.episodes[self._episode_index][2]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        actual_seed = 0 if seed is None else seed
        self._rng = Random(actual_seed + 7919)
        if options and "episode_index" in options:
            self._episode_index = int(options["episode_index"]) % len(self.episodes)
        else:
            self._episode_index = int(self.np_random.integers(len(self.episodes)))
        self._step = 0
        self._queue = []
        self._staffing = self.config.min_agents
        self._last_occupancy = 0.0
        self._last_sla = 1.0
        self._last_abandonment = 0.0
        self._totals = {
            "reward": 0.0,
            "contacts": 0.0,
            "completed": 0.0,
            "abandoned": 0.0,
            "sla_completed": 0.0,
            "wait_sum": 0.0,
            "occupancy_sum": 0.0,
            "agent_hours": 0.0,
            "staffing_sum": 0.0,
            "peak_staffing": 0.0,
            "staffing_changes": 0.0,
        }
        return self._observation(), {"date": self.current_date}

    def _observation(self) -> np.ndarray:
        slot = min(self._step, self.config.horizon - 1)
        forecast = self.forecast_signal
        future = forecast[slot : min(slot + 4, self.config.horizon)]
        angle = 2 * math.pi * slot / self.config.horizon
        values = np.asarray(
            [
                min(len(self._queue) / self.config.backlog_scale, 1.0),
                self._staffing / self.config.max_agents,
                min(float(forecast[slot]) / self.config.demand_scale, 1.0),
                min(float(forecast[min(slot + 1, 47)]) / self.config.demand_scale, 1.0),
                min(float(np.mean(future)) / self.config.demand_scale, 1.0),
                self._last_occupancy,
                self._last_sla,
                self._last_abandonment,
                math.sin(angle),
                math.cos(angle),
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def step(self, action: int):
        policy_action = int(action)
        if not self.action_space.contains(policy_action):
            raise ValueError(f"invalid staffing action: {action}")
        agents = policy_action + self.config.min_agents
        previous_agents = self._staffing
        self._staffing = agents
        arrivals = round(float(self.realized_demand[self._step]))
        self._totals["contacts"] += arrivals
        self._queue.extend((0.0, self.patience_sampler.sample(self._rng)) for _ in range(arrivals))

        capacity_minutes = agents * self.config.interval_minutes
        used_minutes = 0.0
        completed_waits: list[float] = []
        served = 0
        while served < len(self._queue):
            duration = max(0.01, self.service_sampler.sample(self._rng))
            if used_minutes + duration > capacity_minutes:
                break
            age, _ = self._queue[served]
            start_delay = used_minutes / max(agents, 1)
            completed_waits.append(age + start_delay)
            used_minutes += duration
            served += 1
        self._queue = self._queue[served:]

        survivors = []
        abandoned = 0
        for age, patience in self._queue:
            new_age = age + self.config.interval_minutes
            if new_age >= patience:
                abandoned += 1
            else:
                survivors.append((new_age, patience))
        self._queue = survivors

        completed = len(completed_waits)
        sla_completed = sum(wait <= self.config.sla_minutes for wait in completed_waits)
        occupancy = min(used_minutes / max(capacity_minutes, 1), 1.0)
        sla = sla_completed / completed if completed else (1.0 if arrivals == 0 else 0.0)
        abandonment_rate = abandoned / max(arrivals + served + len(self._queue), 1)
        average_wait = sum(completed_waits) / completed if completed else 0.0
        change = abs(agents - previous_agents) / max(self.config.max_agents - 1, 1)
        overload = max(0.0, occupancy - self.config.max_occupancy)
        w = self.reward_weights
        reward = (
            w.service_level * sla
            - w.waiting * min(average_wait / self.config.interval_minutes, 2.0)
            - w.abandonment * abandonment_rate
            - w.staffing * agents / self.config.max_agents
            - w.overload * overload
            - w.staffing_change * change
        )

        self._last_occupancy = occupancy
        self._last_sla = sla
        self._last_abandonment = abandonment_rate
        self._totals["reward"] += reward
        self._totals["completed"] += completed
        self._totals["abandoned"] += abandoned
        self._totals["sla_completed"] += sla_completed
        self._totals["wait_sum"] += sum(completed_waits)
        self._totals["occupancy_sum"] += occupancy
        self._totals["agent_hours"] += agents * self.config.interval_minutes / 60
        self._totals["staffing_sum"] += agents
        self._totals["peak_staffing"] = max(self._totals["peak_staffing"], agents)
        self._totals["staffing_changes"] += int(agents != previous_agents)
        self._step += 1
        terminated = self._step >= self.config.horizon
        info = {
            "date": self.current_date,
            "slot": self._step - 1,
            "agents": agents,
            "arrivals": arrivals,
            "forecast_contacts": float(self.forecast_signal[self._step - 1]),
            "completed": completed,
            "abandoned": abandoned,
            "backlog": len(self._queue),
            "occupancy": occupancy,
            "service_level": sla,
            "average_wait_minutes": average_wait,
            "episode_totals": dict(self._totals),
        }
        return self._observation(), float(reward), terminated, False, info
