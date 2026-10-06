"""Leakage-safe calendar, lag, and rolling demand features."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

LAGS = (1, 2, 48, 96, 336)
ROLLING_WINDOWS = (6, 48, 336)


def calendar_features(timestamp: datetime | pd.Timestamp) -> dict[str, float]:
    slot = timestamp.hour * 2 + timestamp.minute // 30
    weekday = timestamp.weekday()
    return {
        "half_hour_slot": float(slot),
        "day_of_week": float(weekday),
        "is_weekend": float(weekday >= 5),
        "month": float(timestamp.month),
        "slot_sin": math.sin(2 * math.pi * slot / 48),
        "slot_cos": math.cos(2 * math.pi * slot / 48),
        "weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "weekday_cos": math.cos(2 * math.pi * weekday / 7),
    }


def feature_row(timestamp: datetime | pd.Timestamp, history: list[float]) -> dict[str, float]:
    if len(history) < max(LAGS):
        raise ValueError(f"at least {max(LAGS)} historical half-hour slots are required")
    row = calendar_features(timestamp)
    for lag in LAGS:
        row[f"lag_{lag}"] = float(history[-lag])
    values = np.asarray(history, dtype=float)
    for window in ROLLING_WINDOWS:
        prior = values[-window:]
        row[f"rolling_mean_{window}"] = float(prior.mean())
        row[f"rolling_std_{window}"] = float(prior.std(ddof=0))
    return row


def supervised_features(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Build one-step rows; every lag and rolling statistic is shifted into the past."""
    frame = pd.DataFrame(index=series.index)
    for feature_name in calendar_features(series.index[0]):
        frame[feature_name] = [calendar_features(item)[feature_name] for item in series.index]
    for lag in LAGS:
        frame[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window).std(ddof=0)
    valid = frame.notna().all(axis=1)
    return frame.loc[valid].astype(float), series.loc[valid].astype(float)
