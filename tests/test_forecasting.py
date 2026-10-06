from __future__ import annotations

from datetime import timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from callverse.forecasting.baselines import baseline_forecast
from callverse.forecasting.contracts import DemandForecast, ForecastPoint
from callverse.forecasting.data import aggregate_arrivals, chronological_split
from callverse.forecasting.evaluation import calculate_metrics, rolling_origin_evaluate
from callverse.forecasting.features import feature_row, supervised_features
from callverse.forecasting.model import load_artifact
from callverse.forecasting.service import (
    forecast_chart_rows,
    forecast_next_24h,
    forecast_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"
SERIES_PATH = PROJECT_ROOT / "data/processed/forecasting/demand_30min.csv"


def synthetic_series(days: int = 30) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=days * 48, freq="30min")
    values = np.asarray([(item.hour * 2 + item.minute // 30) % 12 for item in index])
    return pd.Series(values, index=index, name="contacts")


def timestamp(value: str):
    return pd.Timestamp(value).to_pydatetime()


def test_aggregation_uses_thirty_minute_slots_and_preserves_counts():
    arrivals = [
        timestamp("2026-01-01 00:01"),
        timestamp("2026-01-01 00:29"),
        timestamp("2026-01-01 01:01"),
    ]
    series = aggregate_arrivals(arrivals)
    assert len(series) == 48
    assert series.iloc[0] == 2
    assert series.iloc[2] == 1


def test_aggregation_generates_zero_count_intervals():
    series = aggregate_arrivals([timestamp("2026-01-01 12:00")])
    assert int((series == 0).sum()) == 47
    assert series.index.to_series().diff().dropna().eq(timedelta(minutes=30)).all()


def test_chronological_split_is_ordered_and_disjoint():
    split = chronological_split(synthetic_series(20))
    assert split.train.index[-1] < split.validation.index[0]
    assert split.validation.index[-1] < split.test.index[0]
    assert len(split.train) + len(split.validation) + len(split.test) == 20 * 48


def test_lag_features_use_only_prior_targets():
    series = synthetic_series(10).astype(float)
    features, target = supervised_features(series)
    timestamp = features.index[0]
    position = series.index.get_loc(timestamp)
    assert features.loc[timestamp, "lag_1"] == series.iloc[position - 1]
    assert features.loc[timestamp, "lag_336"] == series.iloc[position - 336]
    assert target.loc[timestamp] == series.loc[timestamp]


def test_rolling_features_exclude_current_target():
    history = list(range(336))
    row = feature_row(timestamp("2026-01-08"), history)
    assert row["rolling_mean_6"] == np.mean(history[-6:])
    assert row["rolling_mean_48"] == np.mean(history[-48:])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("naive_previous_slot", [399.0, 399.0]),
        ("seasonal_naive_daily", [352.0, 353.0]),
        ("seasonal_naive_weekly", [64.0, 65.0]),
    ],
)
def test_baseline_predictions(name: str, expected: list[float]):
    assert baseline_forecast(list(range(400)), 2, name) == expected


def test_metric_calculations_are_exact_for_perfect_forecast():
    metrics = calculate_metrics([0, 1, 2], [0, 1, 2])
    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert metrics.smape == 0


def test_rolling_origin_evaluates_actual_multi_step_windows():
    series = synthetic_series(10).astype(float)
    result = rolling_origin_evaluate(
        series,
        period_start=series.index[7 * 48],
        period_end=series.index[-1],
        model_name="weekly",
        forecast_function=lambda history, _timestamp, horizon: baseline_forecast(
            history, horizon, "seasonal_naive_weekly"
        ),
    )
    assert result.origins == 3
    assert result.observations == 144
    assert result.overall.mae == 0


def test_final_artifact_loads_and_is_versioned():
    artifact = load_artifact(ARTIFACT_PATH)
    assert artifact.model_name == "lightgbm_poisson"
    assert artifact.model_version == "callverse-demand-v1"
    assert artifact.estimator is not None


def test_forecast_service_returns_exactly_48_non_negative_points():
    frame = pd.read_csv(SERIES_PATH, parse_dates=["timestamp"])
    history = frame.set_index("timestamp")["contacts"]
    forecast = forecast_next_24h(history, ARTIFACT_PATH)
    assert len(forecast.points) == 48
    assert all(point.predicted_contacts >= 0 for point in forecast.points)


def test_forecast_timestamps_progress_by_thirty_minutes():
    frame = pd.read_csv(SERIES_PATH, parse_dates=["timestamp"])
    history = frame.set_index("timestamp")["contacts"]
    points = forecast_next_24h(history, ARTIFACT_PATH).points
    assert all(
        current.timestamp - previous.timestamp == timedelta(minutes=30)
        for previous, current in pairwise(points)
    )


def test_forecast_is_reproducible():
    frame = pd.read_csv(SERIES_PATH, parse_dates=["timestamp"])
    history = frame.set_index("timestamp")["contacts"]
    first = forecast_next_24h(history, ARTIFACT_PATH)
    second = forecast_next_24h(history, ARTIFACT_PATH)
    assert first == second


def test_missing_history_fails_clearly():
    history = synthetic_series(1)
    with pytest.raises(ValueError, match="at least 336"):
        forecast_next_24h(history, ARTIFACT_PATH)


def test_dashboard_forecast_view_model_separates_actual_and_forecast():
    frame = pd.read_csv(SERIES_PATH, parse_dates=["timestamp"])
    history = frame.set_index("timestamp")["contacts"]
    forecast = forecast_next_24h(history, ARTIFACT_PATH)
    rows = forecast_chart_rows(history, forecast)
    assert len(rows) == 96
    assert all(row["Forecast"] is None for row in rows[:48])
    assert all(row["Actual history"] is None for row in rows[48:])


def test_forecast_summary_is_computed_from_points():
    history = synthetic_series(10)
    points = tuple(
        ForecastPoint(
            timestamp=timestamp("2026-02-01") + timedelta(minutes=30 * index),
            predicted_contacts=float(index),
        )
        for index in range(48)
    )
    forecast = DemandForecast(
        forecast_origin=timestamp("2026-01-31 23:30"),
        horizon=48,
        interval_minutes=30,
        points=points,
        model_name="fixture",
        model_version="test",
    )
    summary = forecast_summary(forecast, history)
    assert summary["predicted_total_contacts"] == sum(range(48))
    assert summary["peak_timestamp"] == points[-1].timestamp
    assert summary["peak_contacts"] == 47
