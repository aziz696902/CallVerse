"""Deterministic demand-forecasting baselines."""

from __future__ import annotations

from collections.abc import Sequence

BASELINE_LAGS = {
    "naive_previous_slot": 1,
    "seasonal_naive_daily": 48,
    "seasonal_naive_weekly": 336,
}


def baseline_forecast(history: Sequence[float], horizon: int, model_name: str) -> list[float]:
    if model_name not in BASELINE_LAGS:
        raise ValueError(f"unknown baseline: {model_name}")
    lag = BASELINE_LAGS[model_name]
    if len(history) < lag:
        raise ValueError(f"{model_name} requires at least {lag} historical slots")
    working = [float(value) for value in history]
    predictions: list[float] = []
    for _ in range(horizon):
        prediction = max(0.0, working[-lag])
        predictions.append(prediction)
        working.append(prediction)
    return predictions
