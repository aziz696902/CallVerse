"""Forecasting dataset construction over the existing Technion loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from callverse.calibration.technion import load_technion

INTERVAL_MINUTES = 30


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.Series
    validation: pd.Series
    test: pd.Series

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train.index[-1]

    @property
    def validation_start(self) -> pd.Timestamp:
        return self.validation.index[0]

    @property
    def validation_end(self) -> pd.Timestamp:
        return self.validation.index[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test.index[0]


def aggregate_arrivals(arrivals: list[datetime] | tuple[datetime, ...]) -> pd.Series:
    """Return a continuous, zero-filled 30-minute count series."""
    if not arrivals:
        raise ValueError("at least one arrival timestamp is required")
    timestamps = pd.DatetimeIndex(arrivals).floor("30min")
    start = timestamps.min().normalize()
    end = timestamps.max().normalize() + timedelta(hours=23, minutes=30)
    index = pd.date_range(start=start, end=end, freq="30min")
    counts = pd.Series(1, index=timestamps, dtype="int64").groupby(level=0).sum()
    series = counts.reindex(index, fill_value=0).astype("int64")
    series.name = "contacts"
    series.index.name = "timestamp"
    return series


def load_demand_series(technion_directory: str | Path) -> pd.Series:
    """Reuse the calibration parser and aggregate its accepted arrivals."""
    result = load_technion(technion_directory)
    return aggregate_arrivals(result.arrival_timestamps)


def chronological_split(
    series: pd.Series,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split at midnight boundaries without shuffling."""
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("demand series must have a unique chronological index")
    dates = pd.Index(series.index.normalize().unique())
    if len(dates) < 3:
        raise ValueError("at least three dates are required")
    train_days = int(len(dates) * train_fraction)
    validation_days = int(len(dates) * validation_fraction)
    if min(train_days, validation_days, len(dates) - train_days - validation_days) <= 0:
        raise ValueError("split fractions leave an empty period")
    validation_start = dates[train_days]
    test_start = dates[train_days + validation_days]
    train = series.loc[series.index < validation_start]
    validation = series.loc[(series.index >= validation_start) & (series.index < test_start)]
    test = series.loc[series.index >= test_start]
    return ChronologicalSplit(train=train, validation=validation, test=test)
