"""Application boundary for the next-24-hour demand forecast."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .baselines import BASELINE_LAGS, baseline_forecast
from .contracts import DemandForecast, ForecastPoint
from .features import LAGS
from .model import load_artifact, recursive_lightgbm_forecast

FORECAST_HORIZON = 48


def forecast_next_24h(
    history: pd.Series,
    artifact_path: str | Path,
) -> DemandForecast:
    artifact = load_artifact(artifact_path)
    if history.empty or not history.index.is_monotonic_increasing:
        raise ValueError("chronological demand history is required")
    required = max(LAGS) if artifact.estimator is not None else BASELINE_LAGS.get(artifact.model_name, 0)
    if len(history) < required:
        raise ValueError(f"forecast requires at least {required} half-hour history slots")
    first_timestamp = pd.Timestamp(history.index[-1]) + timedelta(minutes=30)
    values = history.astype(float).tolist()
    if artifact.estimator is None:
        predictions = baseline_forecast(values, FORECAST_HORIZON, artifact.model_name)
    else:
        predictions = recursive_lightgbm_forecast(
            artifact.estimator, values, first_timestamp, FORECAST_HORIZON
        )
    points = tuple(
        ForecastPoint(
            timestamp=(first_timestamp + timedelta(minutes=30 * index)).to_pydatetime(),
            predicted_contacts=float(prediction),
        )
        for index, prediction in enumerate(predictions)
    )
    return DemandForecast(
        forecast_origin=pd.Timestamp(history.index[-1]).to_pydatetime(),
        horizon=FORECAST_HORIZON,
        interval_minutes=30,
        points=points,
        model_name=artifact.model_name,
        model_version=artifact.model_version,
    )


def forecast_summary(forecast: DemandForecast, recent_history: pd.Series) -> dict[str, object]:
    peak = max(forecast.points, key=lambda point: point.predicted_contacts)
    next_two_hours = sum(point.predicted_contacts for point in forecast.points[:4])
    baseline_slots = recent_history.iloc[-48:-44] if len(recent_history) >= 48 else recent_history.iloc[-4:]
    recent_baseline = float(baseline_slots.sum())
    change = None if recent_baseline == 0 else (next_two_hours / recent_baseline - 1) * 100
    high_threshold = max(point.predicted_contacts for point in forecast.points) * 0.8
    high_periods = [
        point.timestamp.isoformat()
        for point in forecast.points
        if point.predicted_contacts >= high_threshold and point.predicted_contacts > 0
    ]
    return {
        "predicted_total_contacts": float(sum(point.predicted_contacts for point in forecast.points)),
        "peak_timestamp": peak.timestamp,
        "peak_contacts": float(peak.predicted_contacts),
        "next_2h_vs_recent_baseline_percent": change,
        "high_demand_periods": high_periods,
    }


def forecast_chart_rows(history: pd.Series, forecast: DemandForecast) -> list[dict[str, object]]:
    rows = [
        {"timestamp": timestamp.to_pydatetime(), "Actual history": float(value), "Forecast": None}
        for timestamp, value in history.iloc[-48:].items()
    ]
    rows.extend(
        {
            "timestamp": point.timestamp,
            "Actual history": None,
            "Forecast": point.predicted_contacts,
        }
        for point in forecast.points
    )
    return rows
