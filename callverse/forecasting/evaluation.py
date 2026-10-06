"""Metrics and rolling-origin 48-step forecast evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)

from .contracts import ForecastEvaluation, ForecastMetrics, HorizonMetrics

ForecastFunction = Callable[[list[float], pd.Timestamp, int], list[float]]


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> ForecastMetrics:
    truth = np.asarray(actual, dtype=float)
    forecast = np.maximum(0.0, np.asarray(predicted, dtype=float))
    if truth.shape != forecast.shape or not truth.size:
        raise ValueError("actual and predicted values must have the same non-empty shape")
    denominator = np.abs(truth) + np.abs(forecast)
    smape_terms = np.divide(
        2 * np.abs(forecast - truth),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    poisson = mean_poisson_deviance(truth, np.maximum(forecast, 1e-9))
    return ForecastMetrics(
        mae=float(mean_absolute_error(truth, forecast)),
        rmse=math.sqrt(float(mean_squared_error(truth, forecast))),
        smape=float(smape_terms.mean() * 100),
        mean_poisson_deviance=float(poisson),
    )

def rolling_origin_evaluate(
    full_series: pd.Series,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    model_name: str,
    forecast_function: ForecastFunction,
    horizon: int = 48,
    stride: int = 48,
) -> ForecastEvaluation:
    positions = range(
        full_series.index.get_loc(period_start),
        full_series.index.get_loc(period_end) - horizon + 2,
        stride,
    )
    actual_windows: list[list[float]] = []
    predicted_windows: list[list[float]] = []
    for position in positions:
        actual = full_series.iloc[position : position + horizon].astype(float).tolist()
        if len(actual) != horizon:
            continue
        history = full_series.iloc[:position].astype(float).tolist()
        predicted = forecast_function(history, full_series.index[position], horizon)
        if len(predicted) != horizon:
            raise ValueError("forecast function returned the wrong horizon")
        actual_windows.append(actual)
        predicted_windows.append(predicted)
    if not actual_windows:
        raise ValueError("no complete rolling-origin windows in evaluation period")

    actual_array = np.asarray(actual_windows)
    predicted_array = np.asarray(predicted_windows)

    def grouped(start: int, stop: int) -> ForecastMetrics:
        return calculate_metrics(actual_array[:, start:stop].ravel(), predicted_array[:, start:stop].ravel())

    return ForecastEvaluation(
        model_name=model_name,
        origins=len(actual_windows),
        observations=int(actual_array.size),
        overall=calculate_metrics(actual_array.ravel(), predicted_array.ravel()),
        by_horizon=HorizonMetrics(
            early_0_to_2h=grouped(0, 4),
            middle_2_to_12h=grouped(4, 24),
            late_12_to_24h=grouped(24, 48),
        ),
    )
